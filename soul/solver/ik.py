"""
Solves the basic IK problem.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
from ..robots.pcc_robot import PCCRobot
from ..costs import pose_cost, limit_cost, world_collision_cost
from ..collision import RobotCollision, CollGeom
from typing import Sequence


def solve_ik(
    robot: PCCRobot,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
    coll: RobotCollision = None,
    world_coll_list: Sequence[CollGeom] = None,
) -> np.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: np.ndarray. Target orientation.
        target_position: np.ndarray. Target position.

    Returns:
        cfg: np.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    if coll is not None and world_coll_list is not None:
        cfg = _solve_ik_jax_with_coll(
            robot,
            coll,
            world_coll_list,
            jnp.array(target_wxyz),
            jnp.array(target_position),
        )
    elif coll is None and world_coll_list is None:
        cfg = _solve_ik_jax(
            robot,
            jnp.array(target_wxyz),
            jnp.array(target_position),
        )
    else:
        raise ValueError("coll and world_coll_list must be either both None or both not None")
    return cfg


@jdc.jit
def _solve_ik_jax(
    robot: PCCRobot,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    robot_var = robot.var_cls(0)
    factors = [
        pose_cost(
            robot,
            robot_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            pos_weight=5.0,
            ori_weight=1.0,
        ),
        limit_cost(
            robot,
            robot_var,
            weight=100.0,
        ),
    ]
    sol, summary = (
        jaxls.LeastSquaresProblem(factors, [robot_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            return_summary=True,
        )
    )
    return sol[robot_var], summary


@jdc.jit
def _solve_ik_jax_with_coll(
    robot: PCCRobot,
    coll: RobotCollision,
    world_coll_list: Sequence[CollGeom],
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    robot_var = robot.var_cls(0)
    factors = [
        pose_cost(
            robot,
            robot_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            pos_weight=5.0,
            ori_weight=1.0,
        ),
        limit_cost(
            robot,
            robot_var,
            weight=100.0,
        ),
    ]
    factors.extend(
        [
            world_collision_cost(
                robot, coll, robot_var, world_coll, 0.05, 5.0
            )
            for world_coll in world_coll_list
        ]
    )
    sol, summary = (
        jaxls.LeastSquaresProblem(factors, [robot_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            return_summary=True,
        )
    )
    return sol[robot_var], summary