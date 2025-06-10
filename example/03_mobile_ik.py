"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time
import viser
import numpy as np
from soul.robots.pcc_robot import PCCRobot
from soul.solver import solve_ik
from soul.collision import HalfSpace, RobotCollision, Sphere
from soul.visualization.visualizer_viser import ViserSoftRobot
from soul.envs.obs_env import ObstacleEnv

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def main():
    robot_config = "configs/robots/pcc_2d_mobile.json"
    map_config = "configs/maps/obstacles.json"

    robot = PCCRobot.from_config(robot_config)
    robot_coll = RobotCollision.from_config(
        robot_config, self_collision_sampling_rate=1
    )
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.8,
        position=(0.0, 0.0, robot.config.length * robot.config.num_sections),
        wxyz=(1, 0, 0, 0),
    )

    env = ObstacleEnv(map_config)
    env.visualize(server, editor=True)
    collision_list = env.get_collision_list()

    while True:
        cfg, _ = solve_ik(
            robot=robot,
            coll=robot_coll,
            world_coll_list=collision_list,
            target_position=ik_target_handle.position,
            target_wxyz=ik_target_handle.wxyz,
        )
        pose = robot.forward_kinematics(cfg)
        robot_vis.update_pose(pose)


if __name__ == "__main__":
    main()
