from __future__ import annotations

import json
import jax
import jax_dataclasses as jdc
import jaxls
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float


@jdc.pytree_dataclass
class CCModelConfig:
    """Configuration for the CCModel."""

    num_sections: jdc.Static[int]
    num_points_per_section: jdc.Static[int]
    length: jdc.Static[float]
    radius: jdc.Static[float]

    # theta_range and phi_range
    lower_limits_theta: Float[Array, "num_sections"]
    upper_limits_theta: Float[Array, "num_sections"]
    lower_limits_phi: Float[Array, "num_sections"]
    upper_limits_phi: Float[Array, "num_sections"]

    opt_mask: jdc.Static[jnp.ndarray]

    @classmethod
    def from_config(cls, config_dict: dict | str) -> CCModelConfig:
        """Creates a config object from a dictionary."""
        if isinstance(config_dict, str):
            config_dict = json.load(open(config_dict))

        opt_mask = (
            config_dict["opt_base_position_mask"]
            + [config_dict["opt_theta"]] * config_dict["num_sections"]
            + [config_dict["opt_phi"]] * config_dict["num_sections"]
        )

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
        )


@jdc.pytree_dataclass
class ConstantCurvatureState:
    """
    State of the CC model (theta, phi per section).
    Length is fixed by CCModelConfig.
    """

    base_position: Float[Array, "3"]
    theta: Float[Array, "num_sections"]  # Curvature for each section
    phi: Float[Array, "num_sections"]  # Rotation angle (phi) for each section

    def __sub__(self, other: ConstantCurvatureState) -> ConstantCurvatureState:
        return ConstantCurvatureState(
            base_position=self.base_position - other.base_position,
            theta=self.theta - other.theta,
            phi=self.phi - other.phi,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConstantCurvatureState):
            return False
        return (
            jnp.allclose(self.base_position, other.base_position, atol=1e-6)
            and jnp.allclose(self.theta, other.theta, atol=1e-6)
            and jnp.allclose(self.phi, other.phi, atol=1e-6)
        )

    def __getitem__(self, indices: Array) -> ConstantCurvatureState:
        return ConstantCurvatureState(
            base_position=self.base_position[indices],
            theta=self.theta[indices],
            phi=self.phi[indices],
        )

    def __len__(self) -> int:
        return self.base_position.shape[0]

    def flatten(self) -> Float[Array, "3 + num_sections + num_sections"]:
        return jnp.concatenate([self.base_position, self.theta, self.phi])

    def repeat(self, n: int, axis: int = 0) -> ConstantCurvatureState:
        # Create tile pattern: all dimensions = 1, except the target axis = n
        def make_tile_pattern(arr_shape, target_axis, repeat_count):
            pattern = [1] * len(arr_shape)
            pattern.insert(target_axis, repeat_count)
            return tuple(pattern)

        return ConstantCurvatureState(
            base_position=jnp.tile(
                jnp.expand_dims(self.base_position, axis=axis),
                make_tile_pattern(self.base_position.shape, axis, n),
            ),
            theta=jnp.tile(
                jnp.expand_dims(self.theta, axis=axis),
                make_tile_pattern(self.theta.shape, axis, n),
            ),
            phi=jnp.tile(
                jnp.expand_dims(self.phi, axis=axis),
                make_tile_pattern(self.phi.shape, axis, n),
            ),
        )
    
    def save_dict(self) -> dict:
        """Saves the state as a dictionary."""
        return {
            "base_position": self.base_position,
            "theta": self.theta,
            "phi": self.phi,
        }

    def load_from_dict(config: dict) -> ConstantCurvatureState:
        """Loads the state from a dictionary."""
        return ConstantCurvatureState(
            base_position=config["base_position"],
            theta=config["theta"],
            phi=config["phi"],
        )


# --- CCModel Class ---
@jdc.pytree_dataclass
class CCRobot:
    """A differentiable Piecewise Constant Curvature (CC) kinematic model."""

    config: CCModelConfig

    var_cls: jdc.Static[type[jaxls.Var[ConstantCurvatureState]]]

    @staticmethod
    def from_config(config_dict: dict) -> CCRobot:

        config = CCModelConfig.from_config(config_dict)

        def retract_fn(
            cfg: ConstantCurvatureState, delta: Array
        ) -> ConstantCurvatureState:
            """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
            delta = delta * config.opt_mask

            return jaxls.Var._euclidean_retract(cfg, delta)

        # do the initial guess, but the value is not important
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

        robot = CCRobot(
            config=config,
            var_cls=StateVar,
        )

        return robot

    @jdc.jit
    def _forward_kinematics(
        self, state: ConstantCurvatureState
    ) -> Float[Array, "num_sections 4 4"]:
        def build_transform(s, p):
            percentage = p / (self.config.num_points_per_section - 1)

            theta = state.theta[s]
            phi = state.phi[s]
            r = self.config.length / theta

            cos_phi = jnp.cos(phi)
            sin_phi = jnp.sin(phi)
            cos_theta = jnp.cos(theta * percentage)
            sin_theta = jnp.sin(theta * percentage)

            Ts_matrix = jnp.array(
                [
                    [
                        cos_phi * cos_phi * (cos_theta - 1.0) + 1.0,
                        sin_phi * cos_phi * (cos_theta - 1.0),
                        cos_phi * sin_theta,
                        r * cos_phi * (1.0 - cos_theta),
                    ],
                    [
                        sin_phi * cos_phi * (cos_theta - 1.0),
                        sin_phi * sin_phi * (cos_theta - 1.0) + 1.0,
                        sin_phi * sin_theta,
                        r * sin_phi * (1.0 - cos_theta),
                    ],
                    [
                        -cos_phi * sin_theta,
                        -sin_phi * sin_theta,
                        cos_theta,
                        r * sin_theta,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            return Ts_matrix

        # Create indices for all segment-point pairs
        segment_indices = jnp.repeat(
            jnp.arange(self.config.num_sections), self.config.num_points_per_section
        )
        point_indices = jnp.tile(
            jnp.arange(self.config.num_points_per_section), self.config.num_sections
        )

        # Vectorize the transform building across all segment-point pairs
        transform_matrices = jax.vmap(build_transform)(segment_indices, point_indices)

        # Process each segment separately using JAX's scan
        final_poses = []
        base_transform = jnp.array(
            [
                [1, 0, 0, state.base_position[0]],
                [0, 1, 0, state.base_position[1]],
                [0, 0, 1, state.base_position[2]],
                [0, 0, 0, 1],
            ]
        )
        prev_pose = jnp.tile(base_transform, (self.config.num_points_per_section, 1, 1))

        for i in range(self.config.num_sections):
            segment_transforms = transform_matrices[
                i
                * self.config.num_points_per_section : (i + 1)
                * self.config.num_points_per_section
            ]
            segment_poses = jnp.matmul(prev_pose, segment_transforms)
            final_poses.append(segment_poses)
            prev_pose = segment_poses[-1]
        # Concatenate all poses
        all_poses = jnp.concatenate(final_poses)
        return all_poses

    def forward_kinematics(
        self, state: ConstantCurvatureState
    ) -> Float[Array, "*batch num_sections 4 4"]:
        if state.theta.ndim == 1:
            return self._forward_kinematics(state)
        else:
            return jax.vmap(self._forward_kinematics)(state)
