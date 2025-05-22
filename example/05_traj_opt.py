"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""
import time
import viser
import numpy as np
from soul.robots.pcc_robot import PCCRobot
from soul.solver import solve_trajopt
from soul.collision import HalfSpace, RobotCollision, Sphere
from soul.visualization.visualizer_viser import ViserSoftRobot

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax
    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)

def main():
    # Setup Environment
    robot = PCCRobot.from_config("configs/pcc_2d.json")
    robot_coll = RobotCollision.from_config("configs/pcc_2d.json")
    server = viser.ViserServer()
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.2])
    )
    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")
    start_handle = server.scene.add_transform_controls(
        "/start", scale=0.3, position=(0.0, 0.0, robot.config.length * robot.config.num_sections), wxyz=(1, 0, 0, 0)
    )
    end_handle = server.scene.add_transform_controls(
        "/end", scale=0.3, position=(0.3, -0.6, 2.5), wxyz=(1, 0, 0, 0)
    )
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.6, position=(0.4, 0.3, 0.4)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())
    server.scene.add_grid("/ground", width=6, height=6)
    plan_button = server.gui.add_button("Plan", disabled=False)
    replay_button = server.gui.add_button("Replay", disabled=False)
    # Set up trajopt parameters
    timesteps = 100
    dt = 0.1
    traj = None

    def plan_callback(args):
        global traj
        sphere_coll_world_current = sphere_coll.transform_from_pos_wxyz(
            position=np.array(sphere_handle.position),
            wxyz=np.array(sphere_handle.wxyz),
        )
        cfg = solve_trajopt(robot, robot_coll, [plane_coll, sphere_coll_world_current], start_handle.position, start_handle.wxyz, end_handle.position, end_handle.wxyz, timesteps, dt)
        traj = robot.forward_kinematics(cfg)
        for i in range(timesteps):
            time.sleep(dt)
            robot_vis.update_pose(traj[i])

    def replay_callback(args):
        global traj
        if traj is None:
            return
        for i in range(timesteps):
            time.sleep(dt)
            robot_vis.update_pose(traj[i])

    plan_button.on_click(plan_callback)
    replay_button.on_click(replay_callback)
    
    while True:
        time.sleep(0.01)

if __name__ == "__main__":
    main()