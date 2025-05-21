from __future__ import annotations

import json

import jax
import jax_dataclasses as jdc
import jaxls
import jaxlie
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float


@jdc.pytree_dataclass
class PCCModelConfig:
    """Configuration for the PCCModel."""

    num_sections: jdc.Static[int]
    num_points_per_section: jdc.Static[int]
    length: jdc.Static[float]

    # kappa_range and phi_range
    lower_limits_kappa: Float[Array, "num_sections"]
    upper_limits_kappa: Float[Array, "num_sections"]
    lower_limits_phi: Float[Array, "num_sections"]
    upper_limits_phi: Float[Array, "num_sections"]

    opt_mask: jdc.Static[jnp.ndarray]

    @classmethod
    def from_config(cls, config_dict: dict | str) -> PCCModelConfig:
        """Creates a config object from a dictionary."""
        if isinstance(config_dict, str):
            config_dict = json.load(open(config_dict))
        
        opt_mask = config_dict["opt_base_position_mask"] + [config_dict["opt_kappa"]] * config_dict["num_sections"] + [config_dict["opt_phi"]] * config_dict["num_sections"]

        return cls(
            num_sections=config_dict["num_sections"],
            num_points_per_section=config_dict["num_points_per_section"],
            length=config_dict["length"],
            opt_mask=jnp.array(opt_mask, dtype=jnp.bool),
            lower_limits_kappa=jnp.array(config_dict["lower_limits_kappa"]),
            upper_limits_kappa=jnp.array(config_dict["upper_limits_kappa"]),
            lower_limits_phi=jnp.array(config_dict["lower_limits_phi"]),
            upper_limits_phi=jnp.array(config_dict["upper_limits_phi"]),
        )


@jdc.pytree_dataclass
class ConstantCurvatureState:
    """
    State of the PCC model (kappa, phi per section).
    Length is fixed by PCCModelConfig.
    """
    base_position: Float[Array, "3"]
    kappa: Float[Array, "num_sections"]  # Curvature for each section
    phi: Float[Array, "num_sections"]  # Rotation angle (phi) for each section

    def to_array(self):
        return jnp.concatenate([self.base_position, self.kappa, self.phi], axis=0)

# --- PCCModel Class ---
@jdc.pytree_dataclass
class PCCRobot:
    """A differentiable Piecewise Constant Curvature (PCC) kinematic model."""

    config: PCCModelConfig

    var_cls: jdc.Static[type[jaxls.Var[ConstantCurvatureState]]]

    @staticmethod
    def from_config(
        config_dict: dict,
        default_cfg: dict | None = None,
    ) -> PCCRobot:

        config = PCCModelConfig.from_config(config_dict)

        def retract_fn(cfg: ConstantCurvatureState, delta: jax.Array) -> ConstantCurvatureState:
            """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
            delta = delta * config.opt_mask
            return jaxls.Var._euclidean_retract(cfg, delta)

        if default_cfg is None:
            default_cfg = ConstantCurvatureState(
                base_position=jnp.zeros(3),
                kappa=jnp.ones((config.num_sections,)),
                phi=jnp.zeros((config.num_sections,)),
            )
        else:
            default_cfg = ConstantCurvatureState.from_dict(default_cfg)

        class StateVar(
            jaxls.Var[Array],
            default_factory=lambda: default_cfg,
            retract_fn=retract_fn,
            tangent_dim=3 + config.num_sections + config.num_sections,
        ): ...

        robot = PCCRobot(
            config=config,
            var_cls=StateVar,
        )

        return robot

    @jdc.jit
    def _forward_kinematics(self, state: ConstantCurvatureState) -> Float[Array, "num_sections 4 4"]:
        def build_transform(s, p):
            kappa = state.kappa[s]
            phi = state.phi[s]
            l = p * self.config.length / (self.config.num_points_per_section - 1)  # scale to [0, length]
            
            cos_phi = jnp.cos(phi)
            sin_phi = jnp.sin(phi)
            cos_kl = jnp.cos(kappa * l)
            sin_kl = jnp.sin(kappa * l)
            
            is_small = jnp.isclose(kappa, 0.0)
            x_trans = jnp.where(is_small, 0.0, cos_phi * (cos_kl - 1.0)/kappa)
            y_trans = jnp.where(is_small, 0.0, sin_phi * (cos_kl - 1.0)/kappa)
            z_trans = jnp.where(is_small, l, sin_kl/kappa)
            
            Ts_matrix = jnp.array([
                [cos_phi * cos_kl, -sin_phi, -cos_phi * sin_kl, x_trans],
                [sin_phi * sin_kl, cos_phi, -sin_phi * sin_kl, y_trans],
                [sin_kl, 0.0, cos_kl, z_trans],
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            return Ts_matrix
        
        # Create indices for all segment-point pairs
        segment_indices = jnp.repeat(jnp.arange(self.config.num_sections), self.config.num_points_per_section)
        point_indices = jnp.tile(jnp.arange(self.config.num_points_per_section), self.config.num_sections)
        
        # Vectorize the transform building across all segment-point pairs
        transform_matrices = jax.vmap(build_transform)(segment_indices, point_indices)
        
        # Process each segment separately using JAX's scan
        final_poses = []
        prev_pose = jnp.tile(jnp.eye(4), (self.config.num_points_per_section, 1, 1))
        
        for i in range(self.config.num_sections):
            segment_transforms = transform_matrices[i*self.config.num_points_per_section:(i+1)*self.config.num_points_per_section]
            segment_poses = jnp.matmul(prev_pose, segment_transforms)
            final_poses.append(segment_poses)
            prev_pose = segment_poses[-1]
        # Concatenate all poses
        all_poses = jnp.concatenate(final_poses)
        return all_poses

    def forward_kinematics(self, state: ConstantCurvatureState) -> Float[Array, "*batch num_sections 4 4"]:
        if state.kappa.ndim == 1:
            return self._forward_kinematics(state)
        else:
            return jax.vmap(self._forward_kinematics)(state)