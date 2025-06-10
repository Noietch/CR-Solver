"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time
import viser
import numpy as np
from soul.robots.pcc_robot import PCCRobot
from soul.solver import solve_ik
from soul.collision import HalfSpace, RobotCollision, Sphere, Capsule
from soul.visualization.visualizer_viser import ViserSoftRobot


DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def main():
    robot = PCCRobot.from_config("configs/robots/pcc_2d_mobile.json")
    robot_coll = RobotCollision.from_config("configs/robots/pcc_2d_mobile.json")
    server = viser.ViserServer()
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )

    obstacle_1 = Capsule.from_radius_height(
        np.array([0.0, 0.0, 0.0]),
        np.array([1]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1, 0, 0, 0]),
    )
    obstacle_2 = Capsule.from_radius_height(
        np.array([0.0, 0.0, 0.0]),
        np.array([1]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([1, 0, 0, 0]),
    )

    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.8,
        position=(0.0, 0.0, robot.config.length * robot.config.num_sections),
        wxyz=(1, 0, 0, 0),
    )

    # obstacle 1
    sphere_handle_1 = server.scene.add_transform_controls(
        "/obstacle/obstacle_1", scale=0.8, position=(0.4, 0.3, 0.4)
    )
    breakpoint()
    server.scene.add_mesh_trimesh(
        "/obstacle/obstacle_1/mesh", mesh=obstacle_1.to_trimesh()
    )

    # obstacle 2
    sphere_handle_2 = server.scene.add_transform_controls(
        "/obstacle/obstacle_2", scale=0.8, position=(0.4, 0.3, 0.4)
    )
    server.scene.add_mesh_trimesh(
        "/obstacle/obstacle_2/mesh", mesh=obstacle_2.to_trimesh()
    )
    server.scene.add_grid("/ground", width=6, height=6)

    # add a slider to control the obstacle 1 position
    obstacle_1_position_handle = server.gui.add_vector3(
        "Obstacle 1 Position", sphere_handle_1.position, disabled=True
    )
    obstacle_2_position_handle = server.gui.add_vector3(
        "Obstacle 2 Position", sphere_handle_2.position, disabled=True
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    target_handle = server.gui.add_vector3(
        "Target", ik_target_handle.position, disabled=True
    )

    while True:
        capsule_coll_world_current = obstacle_1.transform_from_pos_wxyz(
            position=np.array(sphere_handle_1.position),
            wxyz=np.array(sphere_handle_1.wxyz),
        )
        capsule_coll_world_current = obstacle_2.transform_from_pos_wxyz(
            position=np.array(sphere_handle_2.position),
            wxyz=np.array(sphere_handle_2.wxyz),
        )
        start_time = time.time()
        cfg, _ = solve_ik(
            robot=robot,
            coll=robot_coll,
            world_coll_list=[plane_coll, capsule_coll_world_current],
            target_position=ik_target_handle.position,
            target_wxyz=ik_target_handle.wxyz,
        )
        pose = robot.forward_kinematics(cfg)
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        robot_vis.update_pose(pose)


if __name__ == "__main__":
    main()
