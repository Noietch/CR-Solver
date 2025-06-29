from typing import Sequence

import jax
import jax.numpy as jnp
import jaxls
import jaxlie
import numpy as np

from ..robots.pcc_robot import PCCRobot, ConstantCurvatureState
from ..geom import RobotCollision, CollGeom
from ..costs import (
    pose_cost,
    limit_cost,
    smoothness_cost,
    continuous_collision_cost,
    trajectory_length_cost,
    rest_base_cost,
    shape_cost,
)
from .ik_solver import IKSolver


class MotionPlanner:
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
        self._ik_solver_best = jax.jit(self.ik_solver.solve_ik_best_with_coll_start_end)

        self._robot_batch = jax.tree.map(lambda x: x[None], self.robot)
        self._robot_coll_batch = jax.tree.map(lambda x: x[None], self.coll)

    def start_end_interpolate(
        self,
        start_position: np.ndarray,
        start_wxyz: np.ndarray,
        end_position: np.ndarray,
        end_wxyz: np.ndarray,
        world_coll: Sequence[CollGeom],
    ):
        results: ConstantCurvatureState = self._ik_solver_best(
            start_wxyz, start_position, end_wxyz, end_position, world_coll
        )
        base_position = jnp.linspace(
            results[0].base_position, results[1].base_position, self.timesteps
        )
        kappa = jnp.linspace(results[0].kappa, results[1].kappa, self.timesteps)
        phi = jnp.linspace(results[0].phi, results[1].phi, self.timesteps)
        return ConstantCurvatureState(base_position=base_position, kappa=kappa, phi=phi)

    def optimize(
        self,
        init_traj: ConstantCurvatureState,
        world_coll: Sequence[CollGeom],
    ):
        """
        Solves the Trajectory Optimization problem.
        """
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
                    lambda vals, var: ((vals[var] - init_traj[0])).flatten() * 100.0,
                    (self.robot.var_cls(jnp.arange(0, 2)),),
                    name="start_pose_constraint",
                ),
                jaxls.Cost(
                    lambda vals, var: ((vals[var] - init_traj[-1])).flatten() * 100.0,
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
                    jnp.array([20.0])[None],
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


class ConstrainedMotionPlanner(MotionPlanner):

    def tip_traj_follow(self, reference_traj: jaxlie.SE3, world_coll: Sequence[CollGeom]):
        batched_ik_solver = jax.vmap(self.ik_solver.solve_ik_best)
        init_traj = batched_ik_solver(
            reference_traj.wxyz_xyz[..., :4], reference_traj.wxyz_xyz[..., 4:]
        )

        traj_vars = self.robot.var_cls(jnp.arange(self.timesteps))

        # 1. Basic regularization / limit costs.
        factors: list[jaxls.Cost] = [
            pose_cost(
                self._robot_batch,
                traj_vars,
                reference_traj,
                jnp.array([5.0])[None],
                jnp.array([1.0])[None],
            ),
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
            rest_base_cost(
                traj_vars,
                jnp.array([10.0])[None],
            ),
        ]
        # 2. Add start and end pose constraints.
        factors.extend(
            [
                jaxls.Cost(
                    lambda vals, var: ((vals[var] - init_traj[0])).flatten() * 100.0,
                    (self.robot.var_cls(jnp.arange(0, 2)),),
                    name="start_pose_constraint",
                ),
                jaxls.Cost(
                    lambda vals, var: ((vals[var] - init_traj[-1])).flatten() * 100.0,
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
                    jnp.array([50.0])[None],
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
