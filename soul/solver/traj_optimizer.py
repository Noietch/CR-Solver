from typing import Sequence, Optional
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp
import jaxls
import jaxlie
import numpy as np

from ..robots.cc_robot import CCRobot, ConstantCurvatureState
from ..robots.tdcr_robot import TDCRRobot
from ..geom import RobotCollision, CollGeom
from ..costs import (
    pose_cost,
    limit_cost,
    smoothness_cost,
    continuous_collision_cost,
    trajectory_length_cost,
    tendon_length_velocity_cost,
    tendon_length_acceleration_cost,
)
from .ik_solver import IKSolver


@dataclass
class TrajOptimizerOptions:
    """Options for trajectory optimization"""

    # Basic costs weights
    limit_weight: float = 10.0
    smoothness_weight: float = 12.0
    trajectory_length_weight: float = 15.0

    # Collision weight
    collision_weight: float = 40.0

    # Constraint weights
    start_pose_weight: float = 100.0
    end_pose_weight: float = 100.0

    # Time optimization specific weights
    tendon_vel_weight: float = 5.0
    tendon_acc_weight: float = 10.0
    # time_smoothness_weight: float = 8.0
    # time_collision_weight: float = 40.0

    # Trajectory following weights
    pose_position_weight: float = 100.0
    pose_orientation_weight: float = 50.0
    # follow_limit_weight: float = 100.0
    # follow_smoothness_weight: float = 40.0
    # follow_collision_weight: float = 50.0

    # Time step for time optimization
    dt: float = 0.1

    def to_jax_dict(self):
        """Convert options to a dictionary of JAX arrays."""
        return {k: jnp.array(v) for k, v in asdict(self).items()}


class TrajOptimizer:
    def __init__(
        self,
        robot: CCRobot,
        coll: RobotCollision,
        timesteps: int,
        options: Optional[TrajOptimizerOptions] = None,
    ):
        self.timesteps = timesteps
        self.robot = robot
        self.coll = coll
        self.options = options or TrajOptimizerOptions()
        self.ik_solver = IKSolver(
            robot,
            num_seeds_init=10,
            num_seeds_final=1,
            total_steps=64,
            init_steps=6,
            coll=coll,
        )
        self._ik_solver_best = jax.jit(self.ik_solver.solve_ik_best_with_coll_start_end)
        self._batched_ik_solver = jax.vmap(self.ik_solver.solve_ik_best)

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
        theta = jnp.linspace(results[0].theta, results[1].theta, self.timesteps)
        phi = jnp.linspace(results[0].phi, results[1].phi, self.timesteps)
        return ConstantCurvatureState(base_position=base_position, theta=theta, phi=phi)

    def optimize(
        self,
        init_traj: ConstantCurvatureState,
        world_coll: Sequence[CollGeom],
        limit_weight: Optional[float] = None,
        smoothness_weight: Optional[float] = None,
        trajectory_length_weight: Optional[float] = None,
        collision_weight: Optional[float] = None,
        start_pose_weight: Optional[float] = None,
        end_pose_weight: Optional[float] = None,
    ):
        # Use provided weights or fall back to instance options
        limit_w = (
            limit_weight if limit_weight is not None else self.options.limit_weight
        )
        smooth_w = (
            smoothness_weight
            if smoothness_weight is not None
            else self.options.smoothness_weight
        )
        traj_len_w = (
            trajectory_length_weight
            if trajectory_length_weight is not None
            else self.options.trajectory_length_weight
        )
        coll_w = (
            collision_weight
            if collision_weight is not None
            else self.options.collision_weight
        )
        start_w = (
            start_pose_weight
            if start_pose_weight is not None
            else self.options.start_pose_weight
        )
        end_w = (
            end_pose_weight
            if end_pose_weight is not None
            else self.options.end_pose_weight
        )

        traj_vars = self.robot.var_cls(jnp.arange(self.timesteps))

        # 1. Basic regularization / limit costs.
        factors: list[jaxls.Cost] = [
            limit_cost(
                self._robot_batch,
                traj_vars,
                jnp.array([limit_w])[None],
            ),
            smoothness_cost(
                self.robot.var_cls(jnp.arange(1, self.timesteps)),
                self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                jnp.array([smooth_w])[None],
            ),
            trajectory_length_cost(
                self._robot_batch,
                self.robot.var_cls(jnp.arange(1, self.timesteps)),
                self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                jnp.array([traj_len_w])[None],
            ),
        ]
        # 2. Add start and end pose constraints.
        factors.extend(
            [
                jaxls.Cost(
                    lambda vals, var, weight=start_w: (
                        (vals[var] - init_traj[0])
                    ).flatten()
                    * weight,
                    (self.robot.var_cls(jnp.arange(0, 2)),),
                    name="start_pose_constraint",
                ),
                jaxls.Cost(
                    lambda vals, var, weight=end_w: (
                        (vals[var] - init_traj[-1])
                    ).flatten()
                    * weight,
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
                    jnp.array([coll_w])[None],
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

    def optimize_tdcr(
        self,
        init_traj: ConstantCurvatureState,
        world_coll: Sequence[CollGeom],
        limit_weight: Optional[float] = None,
        smoothness_weight: Optional[float] = None,
        trajectory_length_weight: Optional[float] = None,
        collision_weight: Optional[float] = None,
        start_pose_weight: Optional[float] = None,
        end_pose_weight: Optional[float] = None,
        tendon_vel_weight: Optional[float] = None,
        tendon_acc_weight: Optional[float] = None,
        dt: Optional[float] = None,
    ):
        """
        Optimize trajectory with time-based smoothness constraints for tendon lengths.
        This is specifically designed for TDCR robots to ensure smooth tendon movements.

        Args:
            init_traj: Initial trajectory guess
            world_coll: World collision objects
            Various weight parameters for optimization

        Returns:
            Optimized trajectory
        """
        # Use provided weights or fall back to instance options
        limit_w = (
            limit_weight if limit_weight is not None else self.options.limit_weight
        )
        smooth_w = (
            smoothness_weight
            if smoothness_weight is not None
            else self.options.smoothness_weight
        )
        traj_len_w = (
            trajectory_length_weight
            if trajectory_length_weight is not None
            else self.options.trajectory_length_weight
        )
        coll_w = (
            collision_weight
            if collision_weight is not None
            else self.options.collision_weight
        )
        start_w = (
            start_pose_weight
            if start_pose_weight is not None
            else self.options.start_pose_weight
        )
        end_w = (
            end_pose_weight
            if end_pose_weight is not None
            else self.options.end_pose_weight
        )
        tendon_vel_w = (
            tendon_vel_weight
            if tendon_vel_weight is not None
            else self.options.tendon_vel_weight
        )
        tendon_acc_w = (
            tendon_acc_weight
            if tendon_acc_weight is not None
            else self.options.tendon_acc_weight
        )
        dt_val = dt if dt is not None else self.options.dt

        traj_vars = self.robot.var_cls(jnp.arange(self.timesteps))

        # 1. Basic regularization / limit costs
        factors: list[jaxls.Cost] = [
            limit_cost(
                self._robot_batch,
                traj_vars,
                jnp.array([limit_w])[None],
            ),
            smoothness_cost(
                self.robot.var_cls(jnp.arange(1, self.timesteps)),
                self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                jnp.array([smooth_w])[None],
            ),
        ]

        # Tendon length velocity and acceleration smoothness
        factors.extend(
            [
                tendon_length_velocity_cost(
                    self._robot_batch,
                    self.robot.var_cls(jnp.arange(1, self.timesteps)),
                    self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                    jnp.array([dt_val])[None],
                    jnp.array([tendon_vel_w])[None],
                ),
                tendon_length_acceleration_cost(
                    self._robot_batch,
                    self.robot.var_cls(jnp.arange(0, self.timesteps - 2)),
                    self.robot.var_cls(jnp.arange(1, self.timesteps - 1)),
                    self.robot.var_cls(jnp.arange(2, self.timesteps)),
                    jnp.array([dt_val])[None],
                    jnp.array([tendon_acc_w])[None],
                ),
            ]
        )

        # 3. Add trajectory length cost for efficient motion
        factors.append(
            trajectory_length_cost(
                self._robot_batch,
                self.robot.var_cls(jnp.arange(1, self.timesteps)),
                self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                jnp.array([traj_len_w])[None],
            )
        )

        # 4. Add start and end pose constraints
        factors.extend(
            [
                jaxls.Cost(
                    lambda vals, var, weight=start_w: (
                        (vals[var] - init_traj[0])
                    ).flatten()
                    * weight,
                    (self.robot.var_cls(jnp.array([0])),),
                    name="start_pose_constraint",
                ),
                jaxls.Cost(
                    lambda vals, var, weight=end_w: (
                        (vals[var] - init_traj[-1])
                    ).flatten()
                    * weight,
                    (self.robot.var_cls(jnp.array([self.timesteps - 1])),),
                    name="end_pose_constraint",
                ),
            ]
        )

        # 5. Add collision avoidance costs
        for world_coll_obj in world_coll:
            factors.append(
                continuous_collision_cost(
                    self._robot_batch,
                    self._robot_coll_batch,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                    self.robot.var_cls(jnp.arange(1, self.timesteps)),
                    jnp.array([coll_w])[None],
                )
            )

        # 6. Solve the optimization problem
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

    def optimize_tip_traj_follow(
        self,
        reference_traj: jaxlie.SE3,
        world_coll: Sequence[CollGeom],
        options: Optional[TrajOptimizerOptions] = None,
    ):
        # Use provided options or fall back to instance options
        opts = options if options is not None else self.options

        init_traj = self._batched_ik_solver(
            reference_traj.wxyz_xyz[..., :4], reference_traj.wxyz_xyz[..., 4:]
        )

        traj_vars = self.robot.var_cls(jnp.arange(self.timesteps))

        # 1. Basic regularization / limit costs.
        factors: list[jaxls.Cost] = [
            pose_cost(
                self._robot_batch,
                traj_vars,
                reference_traj,
                jnp.array([opts.pose_position_weight])[None],
                jnp.array([opts.pose_orientation_weight])[None],
            ),
            limit_cost(
                self._robot_batch,
                traj_vars,
                jnp.array([opts.limit_weight])[None],
            ),
            smoothness_cost(
                self.robot.var_cls(jnp.arange(1, self.timesteps)),
                self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                jnp.array([opts.smoothness_weight])[None],
            ),
        ]
        # 2. Add collision avoidance costs.
        for world_coll_obj in world_coll:
            factors.append(
                continuous_collision_cost(
                    self._robot_batch,
                    self._robot_coll_batch,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                    self.robot.var_cls(jnp.arange(1, self.timesteps)),
                    jnp.array([opts.collision_weight])[None],
                )
            )
        # 3. Solve the optimization problem.
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
