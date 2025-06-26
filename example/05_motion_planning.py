import jax
import time
import viser
import numpy as np
from soul.robots.pcc_robot import PCCRobot
from soul.geom import RobotCollision, WorldCollision
from soul.solver import MotionPlanner
from soul.visualization.visualizer_viser import ViserSoftRobot, ViserWorld

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def viser_main():
    # Setup Environment
    robot = PCCRobot.from_config("configs/robots/pcc.json")
    robot_coll = RobotCollision.from_config("configs/robots/pcc.json")
    world_coll = WorldCollision.from_config("configs/maps/obstacles_00.json")

    # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")
    robot_vis.create_sphere_visualizations()
    obstacles_vis = ViserWorld(server, world_coll)
    obstacles_vis.create_mesh_visualizations()

    # Setup GUI
    start_handle = server.scene.add_transform_controls(
        "/start",
        scale=0.3,
        position=(1.0, 0.0, 2.5),
        wxyz=(1, 0, 0, 0),
    )
    end_handle = server.scene.add_transform_controls(
        "/end", scale=0.3, position=(0.0, -1.0, 2.5), wxyz=(1, 0, 0, 0)
    )
    plan_button = server.gui.add_button("Plan", disabled=False)
    replay_button = server.gui.add_button("Replay", disabled=False)

    # Set up trajopt parameters
    timesteps = 100
    traj_solver = MotionPlanner(robot, robot_coll, timesteps)
    start_end_interpolate_jit = jax.jit(traj_solver.start_end_interpolate)
    optimize_jit = jax.jit(traj_solver.optimize)

    traj = None

    def plan_callback(args):
        print("Start planning....")
        global traj

        cfg = start_end_interpolate_jit(
            start_handle.position,
            start_handle.wxyz,
            end_handle.position,
            end_handle.wxyz,
            world_coll.collision_geoms,
        )
        cfg = optimize_jit(cfg, world_coll.collision_geoms)
        traj = robot.forward_kinematics(cfg)
        print("Finish planning....")
        robot_vis.visualize_traj_collisions(robot, cfg)
        for i in range(timesteps):
            time.sleep(0.01)
            robot_vis.update_pose(traj[i])

    def replay_callback(args):
        global traj
        if traj is None:
            return
        for i in range(timesteps):
            time.sleep(0.01)
            robot_vis.update_pose(traj[i])

    plan_button.on_click(plan_callback)
    replay_button.on_click(replay_callback)

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    viser_main()
