"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""
import jax
import numpy as np
from soul.robots.pcc_robot import PCCRobot
from soul.solver import solve_ik
from soul.visualization.visualizer import visualize_pcc_model_2d, visualize_pcc_model_3d

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)

def main():
    """Main function for basic IK."""
    robot = PCCRobot.from_config("configs/pcc_2d.json")
    target_wxyz = np.array([1, 0, 0, 0])
    target_position = np.array([0.0, 0.0, 1.8])
    cfg = solve_ik(robot, target_wxyz, target_position)
    pose = robot.forward_kinematics(cfg)
    visualize_pcc_model_2d(pose, target_position=target_position, num_points=robot.config.num_points_per_section, save_path="visualization/ik_result_2d.png")
    visualize_pcc_model_3d(pose, target_position=target_position, num_points=robot.config.num_points_per_section, save_path="visualization/ik_result_3d.png")

if __name__ == "__main__":
    main()
