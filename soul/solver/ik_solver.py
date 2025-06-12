import jax
import jaxls
import jaxlie
import jax.numpy as jnp
import jax_dataclasses as jdc

from ..robots.pcc_robot import PCCRobot, ConstantCurvatureState
from ..solver.utils import roberts_sequence, newton_raphson
from ..solver.ik import pose_cost, limit_cost


class IKSolver:
    def __init__(
        self,
        robot: PCCRobot,
        num_seeds_init: int,
        num_seeds_final: int,
        total_steps: int,
        init_steps: int,
    ):
        self.robot = robot
        self.sample_root = newton_raphson(
            lambda x: x ** (robot.config.num_sections + 1) - x - 1, 1.0, 10_000
        )
        self.num_seeds_init = num_seeds_init
        self.num_seeds_final = num_seeds_final
        self.total_steps = total_steps
        self.init_steps = init_steps

    def sample_states(self, num_states: int) -> ConstantCurvatureState:
        kappa = self.robot.config.lower_limits_kappa + roberts_sequence(
            num_states, self.robot.config.num_sections, self.sample_root
        ) * (
            self.robot.config.upper_limits_kappa - self.robot.config.lower_limits_kappa
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


    def solve_ik_best(self, target_wxyz: jax.Array, target_position: jax.Array) -> jax.Array:
        best_sols, summary = self.solve_ik(target_wxyz, target_position)
        return best_sols[
            jnp.argmin(
                summary.cost_history[
                    jnp.arange(self.num_seeds_final), summary.iterations
                ]
            )
        ]   