import jax
import jaxls
import jaxlie
import jax.numpy as jnp
from typing import Sequence

from ..robots.pcc_robot import PCCRobot, ConstantCurvatureState
from ..solver.utils import roberts_sequence, newton_raphson
from ..costs import (
    pose_cost,
    limit_cost,
    self_collision_cost,
    world_collision_cost,
    smoothness_cost,
)
from ..geom import RobotCollision, CollGeom


class IKSolver:
    def __init__(
        self,
        robot: PCCRobot,
        num_seeds_init: int,
        num_seeds_final: int,
        total_steps: int,
        init_steps: int,
        coll: RobotCollision = None,
    ):
        self.robot = robot
        self.sample_root = newton_raphson(
            lambda x: x ** (robot.config.num_sections + 1) - x - 1, 1.0, 10_000
        )
        self.num_seeds_init = num_seeds_init
        self.num_seeds_final = num_seeds_final
        self.total_steps = total_steps
        self.init_steps = init_steps
        self.coll = coll

    def sample_states(self, num_states: int) -> ConstantCurvatureState:
        kappa = self.robot.config.lower_limits_theta + roberts_sequence(
            num_states, self.robot.config.num_sections, self.sample_root
        ) * (
            self.robot.config.upper_limits_theta - self.robot.config.lower_limits_theta
        )

        phi = self.robot.config.lower_limits_phi + roberts_sequence(
            num_states, self.robot.config.num_sections, self.sample_root
        ) * (self.robot.config.upper_limits_phi - self.robot.config.lower_limits_phi)

        states = ConstantCurvatureState(
            base_position=jnp.zeros((num_states, 3)),
            kappa=kappa,
            phi=phi,
        )
        return states

    def solve_ik(self, target_wxyz: jax.Array, target_position: jax.Array) -> jax.Array:

        def solve_one(
            initial_states: jax.Array, lambda_initial: float | jax.Array, max_iters: int
        ) -> tuple[jax.Array, jaxls.SolveSummary]:
            """Solve IK problem with a single initial condition. We'll vmap
            over initial_states to solve problems in parallel."""
            robot_var = self.robot.var_cls(0)
            factors = [
                pose_cost(
                    self.robot,
                    robot_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz), target_position
                    ),
                    pos_weight=5.0,
                    ori_weight=1.0,
                ),
                limit_cost(
                    self.robot,
                    robot_var,
                    weight=100.0,
                ),
            ]
            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [robot_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make(
                        [robot_var.with_value(initial_states)]
                    ),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
            )
            return sol[robot_var], summary

        vmapped_solve = jax.vmap(solve_one, in_axes=(0, 0, None))

        # Create initial seeds, but this time with quasi-random sequence.
        initial_states = self.sample_states(self.num_seeds_init)

        # Optimize the initial seeds.
        initial_sols, summary = vmapped_solve(
            initial_states, jnp.full(self.num_seeds_init, 10.0), self.init_steps
        )

        # Get the best initial solutions.
        best_initial_sols = jnp.argsort(
            summary.cost_history[jnp.arange(self.num_seeds_init), -1]
        )[: self.num_seeds_final]

        # Optimize more for the best initial solutions.
        best_sols, summary = vmapped_solve(
            initial_sols[best_initial_sols],
            summary.lambda_history[jnp.arange(self.num_seeds_init), -1][
                best_initial_sols
            ],
            self.total_steps - self.init_steps,
        )
        return best_sols, summary

    def solve_ik_best(
        self, target_wxyz: jax.Array, target_position: jax.Array
    ) -> jax.Array:
        best_sols, summary = self.solve_ik(target_wxyz, target_position)
        return best_sols[
            jnp.argmin(
                summary.cost_history[
                    jnp.arange(self.num_seeds_final), summary.iterations
                ]
            )
        ]

    def solve_ik_with_coll(
        self,
        target_wxyz: jax.Array,
        target_position: jax.Array,
        world_coll_list: Sequence[CollGeom],
    ) -> jax.Array:
        def solve_one(
            initial_states: jax.Array, lambda_initial: float | jax.Array, max_iters: int
        ) -> tuple[jax.Array, jaxls.SolveSummary]:
            """Solve IK problem with a single initial condition. We'll vmap
            over initial_states to solve problems in parallel."""
            robot_var = self.robot.var_cls(0)
            factors = [
                pose_cost(
                    self.robot,
                    robot_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz), target_position
                    ),
                    pos_weight=5.0,
                    ori_weight=1.0,
                ),
                limit_cost(
                    self.robot,
                    robot_var,
                    weight=100.0,
                ),
                self_collision_cost(self.robot, self.coll, robot_var, 0.05, 10.0),
            ]
            factors.extend(
                [
                    world_collision_cost(
                        self.robot, self.coll, robot_var, world_coll, 0.05, 10.0
                    )
                    for world_coll in world_coll_list
                ]
            )
            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [robot_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make(
                        [robot_var.with_value(initial_states)]
                    ),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
            )
            return sol[robot_var], summary

        vmapped_solve = jax.vmap(solve_one, in_axes=(0, 0, None))

        # Create initial seeds, but this time with quasi-random sequence.
        initial_states = self.sample_states(self.num_seeds_init)

        # Optimize the initial seeds.
        initial_sols, summary = vmapped_solve(
            initial_states, jnp.full(self.num_seeds_init, 10.0), self.init_steps
        )

        # Get the best initial solutions.
        best_initial_sols = jnp.argsort(
            summary.cost_history[jnp.arange(self.num_seeds_init), -1]
        )[: self.num_seeds_final]

        # Optimize more for the best initial solutions.
        best_sols, summary = vmapped_solve(
            initial_sols[best_initial_sols],
            summary.lambda_history[jnp.arange(self.num_seeds_init), -1][
                best_initial_sols
            ],
            self.total_steps - self.init_steps,
        )
        return best_sols, summary

    def solve_ik_best_with_coll(
        self,
        target_wxyz: jax.Array,
        target_position: jax.Array,
        world_coll_list: Sequence[CollGeom],
    ) -> jax.Array:
        best_sols, summary = self.solve_ik_with_coll(
            target_wxyz, target_position, world_coll_list
        )
        return best_sols[
            jnp.argmin(
                summary.cost_history[
                    jnp.arange(self.num_seeds_final), summary.iterations
                ]
            )
        ]

    def solve_ik_with_coll_start_end(
        self,
        start_wxyz: jax.Array,
        start_position: jax.Array,
        end_wxyz: jax.Array,
        end_position: jax.Array,
        world_coll_list: Sequence[CollGeom],
    ) -> jax.Array:
        def solve_one(
            initial_states: jax.Array, lambda_initial: float | jax.Array, max_iters: int
        ) -> tuple[jax.Array, jaxls.SolveSummary]:
            """Solve IK problem with a single initial condition. We'll vmap
            over initial_states to solve problems in parallel."""
            joint_var_0 = self.robot.var_cls(0)
            joint_var_1 = self.robot.var_cls(1)
            joint_vars = self.robot.var_cls(jnp.arange(2))

            batch_coll = jax.tree.map(lambda x: x[None], self.coll)
            batch_robot = jax.tree.map(lambda x: x[None], self.robot)

            factors = [
                pose_cost(
                    self.robot,
                    joint_var_0,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(start_wxyz), start_position
                    ),
                    pos_weight=5.0,
                    ori_weight=1.0,
                ),
                pose_cost(
                    self.robot,
                    joint_var_1,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(end_wxyz), end_position
                    ),
                    pos_weight=5.0,
                    ori_weight=1.0,
                ),
                limit_cost(
                    batch_robot,
                    joint_vars,
                    jnp.array(100.0)[None],
                ),
                self_collision_cost(batch_robot, batch_coll, joint_vars, 0.05, 10.0),
            ]

            factors.extend(
                [
                    world_collision_cost(
                        batch_robot,
                        batch_coll,
                        joint_vars,
                        jax.tree.map(lambda x: x[None], world_coll),
                        0.05,
                        10.0,
                    )
                    for world_coll in world_coll_list
                ]
            )
            factors.append(
                smoothness_cost(
                    joint_var_0,
                    joint_var_1,
                    jnp.array(1.0),
                )
            )

            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [joint_vars])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make(
                        [joint_vars.with_value(initial_states)]
                    ),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
            )
            return sol[joint_vars], summary

        vmapped_solve = jax.vmap(solve_one, in_axes=(0, 0, None))

        # Create initial seeds, but this time with quasi-random sequence.
        initial_states = self.sample_states(self.num_seeds_init)
        # repeat initial states for start and end
        repeated_initial_states = initial_states.repeat(2, axis=1)

        # Optimize the initial seeds.
        initial_sols, summary = vmapped_solve(
            repeated_initial_states,
            jnp.full(self.num_seeds_init, 10.0),
            self.init_steps,
        )

        # Get the best initial solutions.
        best_initial_sols = jnp.argsort(
            summary.cost_history[jnp.arange(self.num_seeds_init), -1]
        )[: self.num_seeds_final]

        # Optimize more for the best initial solutions.
        best_sols, summary = vmapped_solve(
            initial_sols[best_initial_sols],
            summary.lambda_history[jnp.arange(self.num_seeds_init), -1][
                best_initial_sols
            ],
            self.total_steps - self.init_steps,
        )
        return best_sols, summary

    def solve_ik_best_with_coll_start_end(
        self,
        start_wxyz: jax.Array,
        start_position: jax.Array,
        end_wxyz: jax.Array,
        end_position: jax.Array,
        world_coll_list: Sequence[CollGeom],
    ) -> jax.Array:
        best_sols, summary = self.solve_ik_with_coll_start_end(
            start_wxyz, start_position, end_wxyz, end_position, world_coll_list
        )
        return best_sols[
            jnp.argmin(
                summary.cost_history[
                    jnp.arange(self.num_seeds_final), summary.iterations
                ]
            )
        ]
