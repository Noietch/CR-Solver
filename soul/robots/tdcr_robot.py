from __future__ import annotations

import json
import jax
import jax_dataclasses as jdc
import jaxls
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

from soul.robots.cc_robot import CCRobot, CCModelConfig, ConstantCurvatureState


@jdc.pytree_dataclass
class TDCRModelConfig(CCModelConfig):
    """Configuration for the TDCR Model with tendon drive."""
    
    # Tendon parameters
    disk_radius: jdc.Static[float]
    num_tendons_per_section: jdc.Static[int]
    
    # Tendon positions in local frame (relative to disk center)
    # Shape: (4, num_tendons_per_section) - homogeneous coordinates
    tendon_positions: Float[Array, "4 num_tendons_per_section"]
    
    @classmethod
    def from_config(cls, config_dict: dict | str) -> TDCRModelConfig:
        """Creates a TDCR config object from a dictionary."""
        if isinstance(config_dict, str):
            config_dict = json.load(open(config_dict))
        
        # Get base CC config parameters
        opt_mask = (
            config_dict["opt_base_position_mask"]
            + [config_dict["opt_theta"]] * config_dict["num_sections"]
            + [config_dict["opt_phi"]] * config_dict["num_sections"]
        )
        
        # Set up tendon positions
        num_tendons_per_section = config_dict["num_tendons_per_section"]
        disk_radius = config_dict.get("disk_radius", config_dict.get("radius", 10.0))
        
        # Generate evenly spaced tendon positions around the disk
        angles = jnp.linspace(0, 2 * jnp.pi, num_tendons_per_section, endpoint=False)
        tendon_positions = jnp.array(
            [
                [
                    disk_radius * jnp.cos(angles[i]),
                    disk_radius * jnp.sin(angles[i]),
                    0.0,
                    1.0,
                ]
                for i in range(num_tendons_per_section)
            ]
        ).T
        
        return cls(
            num_sections=config_dict["num_sections"],
            num_points_per_section=config_dict["num_points_per_section"],
            length=config_dict["length"],
            radius=config_dict["radius"],
            opt_mask=jnp.array(opt_mask, dtype=jnp.bool),
            lower_limits_theta=jnp.array(config_dict["lower_limits_theta"]),
            upper_limits_theta=jnp.array(config_dict["upper_limits_theta"]),
            lower_limits_phi=jnp.array(config_dict["lower_limits_phi"]),
            upper_limits_phi=jnp.array(config_dict["upper_limits_phi"]),
            disk_radius=disk_radius,
            num_tendons_per_section=num_tendons_per_section,
            tendon_positions=tendon_positions,
        )


@jdc.pytree_dataclass
class TDCRRobot(CCRobot):
    """Tendon-Driven Continuum Robot with cable length calculation."""
    
    config: TDCRModelConfig
    
    @staticmethod
    def from_config(config_dict: dict) -> TDCRRobot:
        """Create a TDCRRobot from configuration dictionary."""
        
        config = TDCRModelConfig.from_config(config_dict)
        
        def retract_fn(
            cfg: ConstantCurvatureState, delta: Array
        ) -> ConstantCurvatureState:
            """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
            delta = delta * config.opt_mask
            return jaxls.Var._euclidean_retract(cfg, delta)
        
        # Default initial guess
        default_cfg = ConstantCurvatureState(
            base_position=jnp.zeros(3),
            theta=jnp.ones((config.num_sections,)),
            phi=jnp.zeros((config.num_sections,)),
        )
        
        class StateVar(
            jaxls.Var[Array],
            default_factory=lambda: default_cfg,
            retract_fn=retract_fn,
            tangent_dim=3 + config.num_sections + config.num_sections,
        ): ...
        
        robot = TDCRRobot(
            config=config,
            var_cls=StateVar,
        )
        
        return robot
    
    @jdc.jit
    def calculate_tendon_length_single_section(
        self, 
        theta: Float[Array, ""],
        phi: Float[Array, ""],
        section_length: float,
        tendon_position: Float[Array, "4"]  # Homogeneous coordinates
    ) -> Float[Array, ""]:
        """
        Calculate the tendon length for a single section.
        
        Args:
            theta: Curvature angle of the section
            phi: Rotation angle of the section
            section_length: Length of the section
            tendon_position: Homogeneous coordinates (x, y, z, 1) of tendon position
        
        Returns:
            Length of the tendon in this section
        """
        # Extract x, y coordinates from homogeneous coordinates
        x_tendon = tendon_position[0]
        y_tendon = tendon_position[1]
        
        # Distance from tendon to neutral axis
        r_tendon = jnp.sqrt(x_tendon**2 + y_tendon**2)
        
        # Angle of tendon position relative to x-axis
        alpha_tendon = jnp.arctan2(y_tendon, x_tendon)
        
        # Effective bending plane angle relative to tendon
        effective_angle = alpha_tendon - phi
        
        # When theta is very small (straight configuration), use linear approximation
        is_straight = jnp.abs(theta) < 1e-6
        
        # For curved sections
        # The tendon follows a helical path with radius offset from neutral axis
        # Length = section_length * (1 - r_tendon * cos(effective_angle) * theta / section_length)
        curved_length = section_length * (1 - r_tendon * jnp.cos(effective_angle) * theta / section_length)
        
        # For straight sections
        straight_length = section_length
        
        # Select based on whether section is straight
        tendon_length = jnp.where(is_straight, straight_length, curved_length)
        
        return tendon_length
    
    @jdc.jit
    def calculate_tendon_lengths(
        self, 
        state: ConstantCurvatureState
    ) -> Float[Array, "num_sections*num_tendons_per_section"]:
        """
        Calculate the tendon lengths for all tendons given the robot state.
        Each section has its own set of tendons.
        
        Args:
            state: Current state of the robot (base position, theta, phi for each section)
        
        Returns:
            Array of tendon lengths for each tendon in each section
            Shape: (num_sections * num_tendons_per_section,)
        """
        section_length = self.config.length
        
        # Vectorize over sections
        vmap_sections = jax.vmap(
            lambda theta, phi: jax.vmap(
                lambda tendon_pos: self.calculate_tendon_length_single_section(
                    theta, phi, section_length, tendon_pos
                ),
                in_axes=1
            )(self.config.tendon_positions),
            in_axes=0
        )
        
        # Apply vectorized computation
        # Shape: (num_sections, num_tendons_per_section)
        tendon_lengths_2d = vmap_sections(state.theta, state.phi)
        
        # Flatten to 1D array
        tendon_lengths = tendon_lengths_2d.reshape(-1)
        
        return tendon_lengths