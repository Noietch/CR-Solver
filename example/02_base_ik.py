"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time
import viser
import numpy as np
from soul.robots.pcc_robot import PCCRobot
from soul.solver import solve_ik
from soul.visualization.visualizer_plot import (
    visualize_pcc_model_2d,
    visualize_pcc_model_3d,
)
from soul.collision import RobotCollision
from soul.visualization.visualizer_viser import ViserSoftRobot

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def main():
    """Main function for basic IK."""
    robot = PCCRobot.from_config("configs/robots/pcc.json")
    target_wxyz = np.array([0, 0, 1, 0])
    target_position = np.array([0.0, 0.0, 4.0])
    cfg, summary = solve_ik(robot, target_wxyz, target_position)
    print("finish solve ik, final_delta", summary.termination_deltas)
    pose = robot.forward_kinematics(cfg)
    visualize_pcc_model_2d(
        pose,
        target_position=target_position,
        num_points=robot.config.num_points_per_section,
        save_path="visualization/ik_result_2d.png",
    )
    visualize_pcc_model_3d(
        pose,
        target_position=target_position,
        num_points=robot.config.num_points_per_section,
        save_path="visualization/ik_result_3d.png",
    )


def viser_main():
    robot = PCCRobot.from_config("configs/robots/pcc.json")
    robot_coll = RobotCollision.from_config("configs/robots/pcc.json")
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target",
        scale=1,
        position=(0.0, 0.0, robot.config.length * robot.config.num_sections),
        wxyz=(0, 0, 1, 0),
    )
    server.scene.add_grid("/ground", width=6, height=6)
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        start_time = time.time()
        cfg, _ = solve_ik(robot, ik_target_handle.wxyz, ik_target_handle.position)
        pose = robot.forward_kinematics(cfg)
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        robot_vis.update_pose(pose)


if __name__ == "__main__":
    # main()
    viser_main()
