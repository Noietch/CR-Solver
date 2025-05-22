"""
Solves the Trajectory Optimization problem.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np

from ..robots.pcc_robot import PCCRobot, interpolate_states
from ..collision import RobotCollision, CollGeom
from ..costs import (
    limit_cost,
    smoothness_cost,
    continuous_collision_cost,
    boundary_cost,
    start_end_similarity_cost
)
from ..solver import solve_ik_with_collision



def solve_trajopt(
    robot: PCCRobot,
    coll: RobotCollision,
    world_coll_list: Sequence[CollGeom],
    start_position: np.ndarray,
    start_wxyz: np.ndarray,
    end_position: np.ndarray,
    end_wxyz: np.ndarray,
    timesteps: int,
    dt: float,
) -> np.ndarray:
    """
    Solves the Trajectory Optimization problem.
    """
    start_cfg, _ = solve_ik_with_collision(
        robot, coll, world_coll_list, start_position, start_wxyz
    )
    end_cfg, _ = solve_ik_with_collision(
        robot, coll, world_coll_list, end_position, end_wxyz
    )
    init_traj = interpolate_states(start_cfg, end_cfg, timesteps)
    # return init_traj
    traj_vars = robot.var_cls(jnp.arange(timesteps))

    robot = jax.tree.map(lambda x: x[None], robot)  # Add batch dimension.
    robot_coll = jax.tree.map(lambda x: x[None], coll)  # Add batch dimension.

    # Basic regularization / limit costs.
    factors: list[jaxls.Cost] = [
        limit_cost(
            robot,
            traj_vars,
            jnp.array([100.0])[None],
        ),
        smoothness_cost(
            robot.var_cls(jnp.arange(1, timesteps)),
            robot.var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([0.1])[None],
        ),
    ]

    factors.extend(
        [
            jaxls.Cost(
                lambda vals, var: ((vals[var] - start_cfg)).flatten() * 100.0,
                (robot.var_cls(jnp.arange(0, 2)),),
                name="start_pose_constraint",
            ),
            jaxls.Cost(
                lambda vals, var: ((vals[var] - end_cfg)).flatten() * 100.0,
                (robot.var_cls(jnp.arange(timesteps - 2, timesteps)),),
                name="end_pose_constraint",
            ),
        ]
    )
    # Collision avoidance.
    for world_coll_obj in world_coll_list:
        factors.append(
            continuous_collision_cost(
                robot,
                robot_coll,
                jax.tree.map(lambda x: x[None], world_coll_obj),
                robot.var_cls(jnp.arange(0, timesteps - 1)),
                robot.var_cls(jnp.arange(1, timesteps)),
            )
        )
    # 4. Solve the optimization problem.
    solution = (
        jaxls.LeastSquaresProblem(
            factors,
            [traj_vars],
        )
        .analyze()
        .solve(
            # with_verbose=False,
            initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)),
        )
    )
    return solution[traj_vars]



