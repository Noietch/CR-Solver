from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import json
from jaxtyping import Array, Float, Int
from loguru import logger

from ..robots.cc_robot import CCRobot, ConstantCurvatureState
from .collision import collide
from .geometry import Capsule, Sphere, CollGeom


@jdc.pytree_dataclass
class RobotCollision:
    """Collision model for a robot, integrated with robot kinematics."""

    coll: CollGeom

    active_idx_i: Int[Array, "*batch num_active_pairs"]

    active_idx_j: Int[Array, "*batch num_active_pairs"]

    @classmethod
    def from_config(
        cls,
        config_dict: dict | str,
        random_key: int = 0,
        self_collision_sampling_rate: float = 1.0,
        skip_section: int = 2,
    ) -> RobotCollision:
        if isinstance(config_dict, str):
            config_dict = json.load(open(config_dict))

        if "length" in config_dict:
            length = config_dict["length"]
        else:
            length = config_dict["lower_limits_length"]
        num_sections = config_dict["num_sections"]
        num_points_per_section = config_dict["num_points_per_section"]
        radius = config_dict["radius"]

        sphere_list = list[Sphere]()
        length_range = jnp.linspace(
            0, length * num_sections, num_sections * num_points_per_section
        )
        for i in range(num_sections):
            for j in range(num_points_per_section):
                sphere_list.append(
                    Sphere.from_center_and_radius(
                        jnp.array([0, 0, length_range[i * num_points_per_section + j]]),
                        radius,
                    )
                )
        spheres = cast(
            Sphere, jax.tree.map(lambda *args: jnp.stack(args), *sphere_list)
        )
        active_idx_i, active_idx_j = RobotCollision.build_self_collision_pairs(
            num_points_per_section,
            num_sections,
            random_key=random_key,
            sampling_rate=self_collision_sampling_rate,
            skip_section=skip_section,
        )
        robot_coll = cls(
            coll=spheres, active_idx_i=active_idx_i, active_idx_j=active_idx_j
        )
        return robot_coll

    @staticmethod
    def build_self_collision_pairs(
        num_points_per_section: int,
        num_sections: int,
        skip_section: int = 2,
        sampling_rate: int = 0.3,
        random_key: int = 0,
    ) -> None:
        """
        Args:
            num_points_per_section: The number of points per section.
            num_sections: The number of sections.
            skip_section: The number of sections to skip between checking pairs.
        Builds the self-collision pairs for the robot.
        """
        random_key, subkey = jax.random.split(jax.random.PRNGKey(random_key))
        num_points = num_points_per_section * num_sections
        active_idx_i, active_idx_j = [], []
        for i in range(num_points):
            for j in range(num_points):
                is_sampled = jax.random.uniform(subkey) < sampling_rate
                if is_sampled and i + skip_section < j:
                    active_idx_i.append(i)
                    active_idx_j.append(j)
        return jnp.array(active_idx_i), jnp.array(active_idx_j)

    @jdc.jit
    def at_state(self, robot: CCRobot, state: ConstantCurvatureState) -> CollGeom:
        all_poses = robot._forward_kinematics(state)
        all_poses_se3 = jaxlie.SE3.from_matrix(all_poses)
        return self.coll.set_transform(all_poses_se3)

    @jdc.jit
    def get_swept_capsules(
        self,
        robot: CCRobot,
        state_prev: ConstantCurvatureState,
        state_next: ConstantCurvatureState,
    ) -> Capsule:
        coll_prev = self.at_state(robot, state_prev)
        coll_next = self.at_state(robot, state_next)
        swept_capsules = Capsule.from_sphere_pairs(coll_prev, coll_next)
        return swept_capsules

    def compute_self_collision_distance(
        self,
        robot: CCRobot,
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch num_active_pairs"]:
        """
        Computes the signed distances for active self-collision pairs.

        Args:
            robot_coll: The robot's collision model with precomputed active pair indices.
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).

        Returns:
            Signed distances for each active pair.
            Shape: (*batch, num_active_pairs).
            Positive distance means separation, negative means penetration.
        """

        coll = self.at_state(robot, cfg)
        vmapped_collide = jax.vmap(
            jax.vmap(collide, in_axes=(None, 0)), in_axes=(0, None)
        )
        dist_matrix = vmapped_collide(coll, coll)
        active_distances = dist_matrix[..., self.active_idx_i, self.active_idx_j]
        return active_distances

    def compute_world_collision_distance(
        self,
        robot: CCRobot,
        state: ConstantCurvatureState,
        world_geom: CollGeom,  # Shape: (*batch_world, M, ...)
        ignore_prefix: int = 0,
    ) -> Float[Array, "*batch_combined N M"]:
        """
        Computes the signed distances between all robot links (N) and all world obstacles (M).

        Args:
            robot_coll: The robot's collision model.
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).
            world_geom: Collision geometry representing world obstacles. If representing a
                single obstacle, it should have batch shape (). If multiple, the last axis
                is interpreted as the collection of world objects (M).
                The batch dimensions (*batch_world) must be broadcast-compatible with cfg's
                batch axes (*batch_cfg).

        Returns:
            Matrix of signed distances between each robot link and each world object.
            Shape: (*batch_combined, N, M), where N=num_links, M=num_world_objects.
            Positive distance means separation, negative means penetration.
        """
        # 1. Get robot collision geometry at the current config
        # Shape: (*batch_cfg, N, ...)
        coll_robot_world = self.at_state(robot, state)
        N = robot.config.num_sections * robot.config.num_points_per_section
        assert coll_robot_world.get_batch_axes()[-1] == N
        batch_cfg_shape = coll_robot_world.get_batch_axes()[:-1]

        if ignore_prefix > 0:
            coll_robot_world = jax.tree.map(
                lambda x: (
                    x[..., ignore_prefix:, :] if x.ndim >= 2 else x[..., ignore_prefix:]
                ),
                coll_robot_world,
            )

        # 2. Normalize world_geom shape and determine M
        world_axes = world_geom.get_batch_axes()
        if len(world_axes) == 0:  # Single world object
            # Use the object's broadcast_to method to add the M=1 axis correctly
            _world_geom = world_geom.broadcast_to(1)
            M = 1
            batch_world_shape = ()
        else:  # Multiple world objects
            _world_geom = world_geom
            M = world_axes[-1]
            batch_world_shape = world_axes[:-1]

        # 3. Compute distances: Map collide over robot links (axis -2) vs _world_geom (None)
        # _world_geom is guaranteed to have the M axis now.
        _collide_links_vs_world = jax.vmap(collide, in_axes=(-2, None), out_axes=(-2))
        dist_matrix = _collide_links_vs_world(coll_robot_world, _world_geom)

        # 4. Result shape check
        # Calculate expected shape based on broadcasting rules
        expected_batch_combined = jnp.broadcast_shapes(
            batch_cfg_shape, batch_world_shape
        )
        N_filtered = N - ignore_prefix
        expected_shape = (*expected_batch_combined, N_filtered, M)

        # Perform the assertion without try-except or complex logic
        assert dist_matrix.shape == expected_shape, (
            f"Output shape mismatch. Expected {expected_shape}, Got {dist_matrix.shape}. "
            f"Robot axes: {coll_robot_world.get_batch_axes()}, Original World axes: {world_geom.get_batch_axes()}"
        )

        # 5. Return the distance matrix
        return dist_matrix
