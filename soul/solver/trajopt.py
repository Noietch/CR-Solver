"""
Solves the Trajectory Optimization problem.
"""

import functools

import jax
import jax.numpy as jnp
import jaxls
import numpy as np

from ..robots.pcc_robot import PCCRobot, interpolate_states
from ..collision import RobotCollision, CollGeom
from ..costs import (
    limit_cost,
    smoothness_cost,
    continuous_collision_cost,
    trajectory_length_cost,
)
from .ik_solver import IKSolver


class TrajOptSolver:
    def __init__(self, robot: PCCRobot, coll: RobotCollision, timesteps: int):
        self.timesteps = timesteps
        self.robot = robot
        self.coll = coll
        self.ik_solver = IKSolver(
            robot,
            num_seeds_init=10,
            num_seeds_final=1,
            total_steps=64,
            init_steps=6,
            coll=coll,
        )
        self._ik_solver_best = jax.jit(self.ik_solver.solve_ik_best_with_coll)

        self._robot_batch = jax.tree.map(lambda x: x[None], self.robot)
        self._robot_coll_batch = jax.tree.map(lambda x: x[None], self.coll)

    def solve(
        self,
        world_coll: CollGeom,
        start_position: np.ndarray,
        start_wxyz: np.ndarray,
        end_position: np.ndarray,
        end_wxyz: np.ndarray,
    ):
        """
        Solves the Trajectory Optimization problem.
        """
        start_cfg = self._ik_solver_best(start_wxyz, start_position, world_coll)
        end_cfg = self._ik_solver_best(end_wxyz, end_position, world_coll)
        init_traj = interpolate_states(start_cfg, end_cfg, self.timesteps)
        # return init_traj
        traj_vars = self.robot.var_cls(jnp.arange(self.timesteps))

        # 1. Basic regularization / limit costs.
        factors: list[jaxls.Cost] = [
            limit_cost(
                self._robot_batch,
                traj_vars,
                jnp.array([100.0])[None],
            ),
            smoothness_cost(
                self.robot.var_cls(jnp.arange(1, self.timesteps)),
                self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                jnp.array([10.0])[None],
            ),
            trajectory_length_cost(
                self._robot_batch,
                self.robot.var_cls(jnp.arange(1, self.timesteps)),
                self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                jnp.array([10.0])[None],
            ),
        ]
        # 2. Add start and end pose constraints.
        factors.extend(
            [
                jaxls.Cost(
                    lambda vals, var: ((vals[var] - start_cfg)).flatten() * 100.0,
                    (self.robot.var_cls(jnp.arange(0, 2)),),
                    name="start_pose_constraint",
                ),
                jaxls.Cost(
                    lambda vals, var: ((vals[var] - end_cfg)).flatten() * 100.0,
                    (
                        self.robot.var_cls(
                            jnp.arange(self.timesteps - 2, self.timesteps)
                        ),
                    ),
                    name="end_pose_constraint",
                ),
            ]
        )
        # 3. Add collision avoidance costs.
        for world_coll_obj in world_coll:
            factors.append(
                continuous_collision_cost(
                    self._robot_batch,
                    self._robot_coll_batch,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                    self.robot.var_cls(jnp.arange(1, self.timesteps)),
                )
            )
        # 5. Solve the optimization problem.
        solution = (
            jaxls.LeastSquaresProblem(
                factors,
                [traj_vars],
            )
            .analyze()
            .solve(
                verbose=False,
                initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)),
            )
        )
        return solution[traj_vars]
