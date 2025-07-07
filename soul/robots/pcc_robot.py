from __future__ import annotations

import json
import jax
import jax_dataclasses as jdc
import jaxls
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float


@jdc.pytree_dataclass
class PCCModelConfig:
    """Configuration for the PCC (Piecewise Constant Curvature) Model with tendon drive."""

    num_sections: jdc.Static[int]
    num_points_per_section: jdc.Static[int]
    length: jdc.Static[float]  # Length of each section

    # Robot physical parameters
    disk_radius: jdc.Static[float]
    num_tendons: jdc.Static[int]

    # Tendon positions in local frame (relative to disk center)
    tendon_positions: Float[Array, "4 num_tendons"]  # Homogeneous coordinates

    # Optimization parameters
    opt_mask: jdc.Static[jnp.ndarray]  # Which parameters to optimize

    @classmethod
    def from_config(cls, config_dict: dict | str) -> PCCModelConfig:
        """Creates a config object from a dictionary."""
        if isinstance(config_dict, str):
            config_dict = json.load(open(config_dict))

        # Build optimization mask: [base_pos(3) + kappa_x + kappa_y + epsilon] * num_sections
        opt_base_position_mask = config_dict["opt_base_position_mask"]
        opt_kappa_x = config_dict["opt_kappa_x"]
        opt_kappa_y = config_dict["opt_kappa_y"]
        opt_epsilon = config_dict["opt_epsilon"]
        opt_mask = (
            opt_base_position_mask
            + [opt_kappa_x] * config_dict["num_sections"]
            + [opt_kappa_y] * config_dict["num_sections"]
            + [opt_epsilon] * config_dict["num_sections"]
        )

        # Set up base position mask
        num_tendons = config_dict["num_tendons"]
        disk_radius = config_dict["radius"]
        angles = jnp.linspace(0, 2 * jnp.pi, num_tendons, endpoint=False)

        tendon_pos = jnp.array(
            [
                [
                    disk_radius * jnp.cos(angles[i]),
                    disk_radius * jnp.sin(angles[i]),
                    0.0,
                    1.0,
                ]
                for i in range(num_tendons)
            ]
        ).T

        return cls(
            num_sections=config_dict["num_sections"],
            num_points_per_section=config_dict["num_points_per_section"],
            length=jnp.array(config_dict["length"]),
            disk_radius=disk_radius,
            num_tendons=num_tendons,
            tendon_positions=tendon_pos,
            opt_mask=jnp.array(opt_mask, dtype=jnp.bool),
        )


class TendonState:
    """State of the tendon drive."""

    tendon_lengths: Float[Array, "num_tendons"]


@jdc.pytree_dataclass
class PCCState:
    """
    State of the PCC model with tendon drive.
    Each section has kappa_x, kappa_y (curvatures) and epsilon (twist).
    """

    base_position: Float[Array, "3"]
    kappa_x: Float[Array, "num_sections"]  # Curvature in x direction
    kappa_y: Float[Array, "num_sections"]  # Curvature in y direction
    epsilon: Float[Array, "num_sections"]  # Twist angle for each section

    def __sub__(self, other: PCCState) -> PCCState:
        return PCCState(
            base_position=self.base_position - other.base_position,
            kappa_x=self.kappa_x - other.kappa_x,
            kappa_y=self.kappa_y - other.kappa_y,
            epsilon=self.epsilon - other.epsilon,
        )

    def __getitem__(self, indices: Array) -> PCCState:
        return PCCState(
            base_position=self.base_position[indices],
            kappa_x=self.kappa_x[indices],
            kappa_y=self.kappa_y[indices],
            epsilon=self.epsilon[indices],
        )

    def __len__(self) -> int:
        return self.base_position.shape[0]

    def flatten(self) -> Float[Array, "3 + 3*num_sections"]:
        return jnp.concatenate(
            [self.base_position, self.kappa_x, self.kappa_y, self.epsilon]
        )

    def repeat(self, n: int, axis: int = 0) -> PCCState:
        def make_tile_pattern(arr_shape, target_axis, repeat_count):
            pattern = [1] * len(arr_shape)
            pattern.insert(target_axis, repeat_count)
            return tuple(pattern)

        return PCCState(
            base_position=jnp.tile(
                jnp.expand_dims(self.base_position, axis=axis),
                make_tile_pattern(self.base_position.shape, axis, n),
            ),
            kappa_x=jnp.tile(
                jnp.expand_dims(self.kappa_x, axis=axis),
                make_tile_pattern(self.kappa_x.shape, axis, n),
            ),
            kappa_y=jnp.tile(
                jnp.expand_dims(self.kappa_y, axis=axis),
                make_tile_pattern(self.kappa_y.shape, axis, n),
            ),
            epsilon=jnp.tile(
                jnp.expand_dims(self.epsilon, axis=axis),
                make_tile_pattern(self.epsilon.shape, axis, n),
            ),
        )


@jdc.pytree_dataclass
class PCCRobot:
    """A differentiable Piecewise Constant Curvature (PCC) kinematic model with tendon drive."""

    config: PCCModelConfig
    var_cls: jdc.Static[type[jaxls.Var[PCCState]]]

    @staticmethod
    def from_config(config_dict: dict) -> PCCRobot:
        config = PCCModelConfig.from_config(config_dict)

        def retract_fn(cfg: PCCState, delta: Array) -> PCCState:
            """Retraction function with masking for optimization."""
            delta = delta * config.opt_mask
            return jaxls.Var._euclidean_retract(cfg, delta)

        # Default initial state
        default_cfg = PCCState(
            base_position=jnp.zeros(3),
            kappa_x=jnp.zeros((config.num_sections,)),
            kappa_y=jnp.zeros((config.num_sections,)),
            epsilon=jnp.zeros((config.num_sections,)),
        )

        class StateVar(
            jaxls.Var[Array],
            default_factory=lambda: default_cfg,
            retract_fn=retract_fn,
            tangent_dim=3 + 3 * config.num_sections,
        ): ...

        robot = PCCRobot(
            config=config,
            var_cls=StateVar,
        )

        return robot

    @jdc.jit
    def _build_transformation_matrix(
        self, kappa_x: float, kappa_y: float, epsilon: float, length: float
    ) -> Float[Array, "4 4"]:
        """Build transformation matrix for a single section."""
        # Calculate total curvature and bending plane angle
        kappa_x = kappa_x + 1e-6
        kappa_y = kappa_y + 1e-6
        kappa = jnp.sqrt(kappa_x**2 + kappa_y**2)
        phi = jnp.arctan2(kappa_y, kappa_x)

        theta = kappa * length  # Total bending angle
        # Position of the end point
        p_x = (1 - jnp.cos(theta)) * jnp.cos(phi) / kappa
        p_y = (1 - jnp.cos(theta)) * jnp.sin(phi) / kappa
        p_z = jnp.sin(theta) / kappa

        # Rotation matrices
        Rz_phi = jnp.array(
            [
                [jnp.cos(phi), -jnp.sin(phi), 0],
                [jnp.sin(phi), jnp.cos(phi), 0],
                [0, 0, 1],
            ]
        )

        Ry_theta = jnp.array(
            [
                [jnp.cos(theta), 0, jnp.sin(theta)],
                [0, 1, 0],
                [-jnp.sin(theta), 0, jnp.cos(theta)],
            ]
        )

        Rz_epsilon_phi = jnp.array(
            [
                [jnp.cos(epsilon - phi), -jnp.sin(epsilon - phi), 0],
                [jnp.sin(epsilon - phi), jnp.cos(epsilon - phi), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation
        R = Rz_phi @ Ry_theta @ Rz_epsilon_phi

        return jnp.array(
            [
                [R[0, 0], R[0, 1], R[0, 2], p_x],
                [R[1, 0], R[1, 1], R[1, 2], p_y],
                [R[2, 0], R[2, 1], R[2, 2], p_z],
                [0, 0, 0, 1],
            ]
        )

    @jdc.jit
    def _forward_kinematics(self, state: PCCState) -> Float[Array, "total_points 4 4"]:
        """Compute forward kinematics for all points along the robot."""
        # Base transform
        base_transform = jnp.array(
            [
                [1, 0, 0, state.base_position[0]],
                [0, 1, 0, state.base_position[1]],
                [0, 0, 1, state.base_position[2]],
                [0, 0, 0, 1],
            ]
        )

        all_poses = []
        current_base = base_transform

        # Process each section sequentially (for correct cumulative transforms)
        for section_idx in range(self.config.num_sections):
            # Section parameters
            kappa_x = state.kappa_x[section_idx]
            kappa_y = state.kappa_y[section_idx]
            epsilon = state.epsilon[section_idx]

            # Vectorized computation for all points in this section
            point_indices = jnp.arange(self.config.num_points_per_section)
            s_values = point_indices / (self.config.num_points_per_section - 1)
            point_lengths = s_values * self.config.length

            # Use vmap to compute all local transforms for this section
            vmap_build_transform = jax.vmap(
                lambda length: self._build_transformation_matrix(
                    kappa_x, kappa_y, epsilon, length
                )
            )
            local_transforms = vmap_build_transform(point_lengths)

            # Apply current base transform to all local transforms using vmap
            vmap_matmul = jax.vmap(lambda local: current_base @ local)
            section_poses = vmap_matmul(local_transforms)

            all_poses.append(section_poses)

            # Update base transform for next section
            section_end_transform = self._build_transformation_matrix(
                kappa_x, kappa_y, epsilon, self.config.length
            )
            current_base = current_base @ section_end_transform

        # Concatenate all section poses
        return jnp.concatenate(all_poses, axis=0)

    def forward_kinematics(
        self, state: PCCState
    ) -> Float[Array, "*batch total_points 4 4"]:
        """Compute forward kinematics with support for batched states."""
        if state.kappa_x.ndim == 1:
            return self._forward_kinematics(state)
        else:
            return jax.vmap(self._forward_kinematics)(state)

    @jdc.jit
    def compute_tendon_lengths(self, state: PCCState) -> Float[Array, "num_tendons"]:
        """Compute the length of each tendon given the robot state."""
        poses = self._forward_kinematics(state)

        def compute_single_tendon_length(tendon_idx):
            """Compute length for a single tendon."""
            # Get tendon position in local frame
            tendon_local = self.config.tendon_positions[:, tendon_idx]

            # Transform to global positions for all points using vmap
            global_positions = jax.vmap(lambda pose: pose @ tendon_local)(poses)[:, :3]

            # Prepend initial position
            initial_pos = (
                self.config.tendon_positions[:3, tendon_idx] + state.base_position
            )
            all_positions = jnp.concatenate([initial_pos[None, :], global_positions])

            # Compute cumulative length
            segment_lengths = jnp.linalg.norm(
                jnp.diff(all_positions, axis=0) + 1e-6, axis=1
            )
            return jnp.sum(segment_lengths)

        # Vectorize over all tendons
        return jax.vmap(compute_single_tendon_length)(
            jnp.arange(self.config.num_tendons)
        )
