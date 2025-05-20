"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""
import numpy as np
from soul.robots.pcc_robot import PCCRobot
from soul.robots.pcc_robot_array import PCCRobot as PCCRobotArray
from soul.solver import solve_ik
from soul.visualization.visualizer import visualize_pcc_model_2d, visualize_pcc_model_3d



def main():
    """Main function for basic IK."""
    robot = PCCRobot.from_config("configs/pcc_2d.json")
    target_wxyz = np.array([1, 0, 0, 0])
    target_position = np.array([0.0, 0.0, 2.8])
    cfg, summary = solve_ik(robot, target_wxyz, target_position)
    print("finish solve ik, final_delta", summary.termination_deltas)
    pose = robot.forward_kinematics(cfg)
    robot_array = PCCRobotArray.from_config("configs/pcc_2d.json")
    pose = robot_array.forward_kinematics(cfg.to_array())
    visualize_pcc_model_2d(pose, target_position=target_position, num_points=robot_array.config.num_points_per_section, save_path="visualization/ik_result_2d.png")
    visualize_pcc_model_3d(pose, target_position=target_position, num_points=robot_array.config.num_points_per_section, save_path="visualization/ik_result_3d.png")

if __name__ == "__main__":
    main()
