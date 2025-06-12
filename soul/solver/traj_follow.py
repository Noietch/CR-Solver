import jax
import numpy as np
from ..robots.pcc_robot import PCCRobot
from .ik_solver import IKSolver

def solve_ee_traj_follow(
    robot: PCCRobot,
    ee_position: np.ndarray,
    ee_wxyz: np.ndarray,
) -> np.ndarray:
    solver = IKSolver(
        robot, num_seeds_init=64, num_seeds_final=4, total_steps=16, init_steps=6
    )
    batched_ik_solve = jax.vmap(jax.jit(solver.solve_ik))
    solution = batched_ik_solve(ee_wxyz, ee_position)
    # fk_result = robot.forward_kinematics(solution)
    return solution