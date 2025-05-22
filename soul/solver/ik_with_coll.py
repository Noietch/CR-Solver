"""
Solves the basic IK problem with collision avoidance.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np

from ..robots.pcc_robot import PCCRobot
from ..collision import RobotCollision, CollGeom
from ..costs import pose_cost, limit_cost, world_collision_cost, elastic_energy_cost

def solve_ik_with_collision(
    robot: PCCRobot,
    coll: RobotCollision,
    world_coll_list: Sequence[CollGeom],
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
) -> np.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: Sequence[str]. Length: num_targets.
        position: ArrayLike. Shape: (num_targets, 3), or (3,).
        wxyz: ArrayLike. Shape: (num_targets, 4), or (4,).

    Returns:
        cfg: ArrayLike. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)

    T_world_targets = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    cfg = _solve_ik_with_collision_jax(
        robot,
        coll,
        world_coll_list,
        T_world_targets,
    )

    return cfg


@jdc.jit
def _solve_ik_with_collision_jax(
    robot: PCCRobot,
    coll: RobotCollision,
    world_coll_list: Sequence[CollGeom],
    T_world_target: jaxlie.SE3,
) -> jax.Array:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    robot_var = robot.var_cls(0)
    vars = [robot_var]

    # Weights and margins defined directly in factors
    costs = [
        pose_cost(
            robot,
            robot_var,
            target_pose=T_world_target,
            pos_weight=5.0,
            ori_weight=0.0,
        ),
        limit_cost(
            robot,
            robot_var=robot_var,
            weight=100.0,
        ),
        elastic_energy_cost(
            robot_var=robot_var,
            weight=0.5,
        ),
    ]
    costs.extend(
        [
            world_collision_cost(
                robot, coll, robot_var, world_coll, 0.05, 10.0
            )
            for world_coll in world_coll_list
        ]
    )

    sol, summary = (
        jaxls.LeastSquaresProblem(costs, vars)
        .analyze()
        .solve(verbose=False, linear_solver="dense_cholesky", return_summary=True,)
    )
    return sol[robot_var], summary
