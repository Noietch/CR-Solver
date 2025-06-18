import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Optional

from ..robots.pcc_robot import PCCRobot, ConstantCurvatureState
from ..collision.pcc_robot_collision import RobotCollision
from ..collision.geometry import CollGeom
from .ik_solver import IKSolver


def compute_distances(
    states: jnp.ndarray,
    weight: float = 1.0,
) -> jnp.ndarray:
    # compute distances between two timesteps
    states_curr = states[1:, None, :, :]  # [ntimesteps-1, 1, k, state_dim]
    states_prev = states[:-1, :, None, :]  # [ntimesteps-1, k, 1, state_dim]
    return jnp.linalg.norm(states_curr - states_prev, axis=-1) * weight


def compute_state_distances(
    states: ConstantCurvatureState,
    curvature_weight: float = 1.0,
    phi_weight: float = 1.0,
    position_weight: float = 1.0,
) -> jnp.ndarray:
    # compute distances between states
    kappa_distances = compute_distances(states.kappa, weight=curvature_weight)
    phi_distances = compute_distances(states.phi, weight=phi_weight)
    position_distances = compute_distances(states.base_position, weight=position_weight)
    return kappa_distances + phi_distances + position_distances


def solve_ee_traj_follow(
    robot: PCCRobot,
    ee_position: np.ndarray,
    ee_wxyz: np.ndarray,
) -> np.ndarray:
    solver = IKSolver(
        robot, num_seeds_init=64, num_seeds_final=4, total_steps=16, init_steps=6
    )
    batched_ik_solve = jax.vmap(jax.jit(solver.solve_ik_best))
    solution = batched_ik_solve(ee_wxyz, ee_position)
    return solution


def solve_ee_traj_follow_dp(
    robot: PCCRobot,
    ee_position: np.ndarray,
    ee_wxyz: np.ndarray,
) -> np.ndarray:
    solver = IKSolver(
        robot, num_seeds_init=200, num_seeds_final=30, total_steps=16, init_steps=6
    )
    batched_ik_solve = jax.vmap(jax.jit(solver.solve_ik))
    solution, _ = batched_ik_solve(ee_wxyz, ee_position)
    state_distances = compute_state_distances(solution)  # [ntimestep, k, k]

    # dp search
    ntimestep, k, _ = state_distances.shape
    costs = jnp.zeros((ntimestep, k))
    memo = jnp.zeros((ntimestep, k), dtype=jnp.int32)
    for t in range(1, ntimestep):
        t_next_cost = jnp.maximum(state_distances[t - 1, :, :], costs[t - 1, :])
        costs = costs.at[t, :].set(jnp.min(t_next_cost, axis=1))
        memo = memo.at[t, :].set(jnp.argmin(t_next_cost, axis=1))  # [ntimestep, k]

    # backtrack
    best_solution_kappa = jnp.zeros((ntimestep, robot.config.num_sections))
    best_solution_phi = jnp.zeros((ntimestep, robot.config.num_sections))
    best_solution_position = jnp.zeros((ntimestep, 3))
    i = jnp.argmin(costs[-1, :])
    for t in range(ntimestep - 1, -1, -1):
        best_solution_kappa = best_solution_kappa.at[t, :].set(solution[t, i, :].kappa)
        best_solution_phi = best_solution_phi.at[t, :].set(solution[t, i, :].phi)
        best_solution_position = best_solution_position.at[t, :].set(
            solution[t, i, :].base_position
        )
        i = memo[t, i]
    return ConstantCurvatureState(
        base_position=best_solution_position,
        kappa=best_solution_kappa,
        phi=best_solution_phi,
    )
