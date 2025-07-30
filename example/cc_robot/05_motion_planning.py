import jax
import time
import viser
import numpy as np
from soul.robots.cc_robot import CCRobot
from soul.geom import RobotCollision, WorldCollision
from soul.solver import MotionPlanner, SamplingBasedMotionPlanner, RRTMotionPlanner
from soul.visualization.visualizer_viser import ViserSoftRobot, ViserWorld

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def viser_main_trajopt():
    # Setup Environment
    robot = CCRobot.from_config("configs/robots/cc_scene_eval.json")
    robot_coll = RobotCollision.from_config("configs/robots/cc_scene_eval.json")
    world_coll = WorldCollision.from_config(
        "configs/maps/mp_scene/obstacles_13.pick_from_shelf.json"
    )

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
        position=(-0.3, -1.26, 2.51),
        wxyz=(1, 0, 0, 0),
    )
    end_handle = server.scene.add_transform_controls(
        "/end", scale=0.3, position=(-0.4, 1.45, 0.89), wxyz=(1, 0, 0, 0)
    )

    with server.gui.add_folder("Handles Tfs"):
        start_pos_text = server.gui.add_text(
            "Start Pos",
            initial_value=str(tuple(np.round(start_handle.position, 2))),
            disabled=True,
        )
        start_wxyz_text = server.gui.add_text(
            "Start wxyz",
            initial_value=str(tuple(np.round(start_handle.wxyz, 2))),
            disabled=True,
        )
        end_pos_text = server.gui.add_text(
            "End Pos",
            initial_value=str(tuple(np.round(end_handle.position, 2))),
            disabled=True,
        )
        end_wxyz_text = server.gui.add_text(
            "End wxyz",
            initial_value=str(tuple(np.round(end_handle.wxyz, 2))),
            disabled=True,
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

    def on_handle_update(handle: viser.TransformControlsHandle):
        """Update GUI when handles are moved."""
        start_pos_text.value = str(np.round(start_handle.position, 2))
        start_wxyz_text.value = str(np.round(start_handle.wxyz, 2))
        end_pos_text.value = str(np.round(end_handle.position, 2))
        end_wxyz_text.value = str(np.round(end_handle.wxyz, 2))

    plan_button.on_click(plan_callback)
    replay_button.on_click(replay_callback)
    start_handle.on_update(on_handle_update)
    end_handle.on_update(on_handle_update)
    on_handle_update(start_handle)

    while True:
        time.sleep(0.01)


def viser_main_prm():
    # Setup Environment
    robot = CCRobot.from_config("configs/robots/cc.json")
    robot_coll = RobotCollision.from_config("configs/robots/cc.json")
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
    traj_solver = SamplingBasedMotionPlanner(robot, robot_coll, timesteps)
    # find_path_jit = jax.jit(traj_solver.find_path)
    # optimize_jit = jax.jit(traj_solver.optimize)

    traj = None

    def plan_callback(args):
        print("Start planning....")
        global traj

        cfg = traj_solver._ik_solver_best(
            start_handle.wxyz,
            start_handle.position,
            end_handle.wxyz,
            end_handle.position,
            world_coll.collision_geoms,
        )
        cfg = traj_solver.find_path(
            cfg[0],
            cfg[1],
            1000,
            world_coll.collision_geoms,
        )
        if cfg is None:
            print("No solution")
            return
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
        # plan_callback(None)
        time.sleep(0.01)


def viser_main_rrt():
    # Setup Environment
    robot = CCRobot.from_config("configs/robots/cc.json")
    robot_coll = RobotCollision.from_config("configs/robots/cc.json")
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
    traj_solver = RRTMotionPlanner(robot, robot_coll, timesteps)
    # find_path_jit = jax.jit(traj_solver.find_path)
    # optimize_jit = jax.jit(traj_solver.optimize)

    traj = None

    def plan_callback(args):
        print("Start planning....")
        global traj

        cfg = traj_solver._ik_solver_best(
            start_handle.wxyz,
            start_handle.position,
            end_handle.wxyz,
            end_handle.position,
            world_coll.collision_geoms,
        )
        cfg = traj_solver.find_path(
            cfg[0],
            cfg[1],
            world_coll.collision_geoms,
        )
        if cfg is None:
            print("No solution")
            return
        traj = robot.forward_kinematics(cfg)
        print("Finish planning....")
        # robot_vis.visualize_traj_collisions(robot, cfg)
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
        # plan_callback(None)
        time.sleep(0.01)


if __name__ == "__main__":
    viser_main_trajopt()
    # viser_main_prm()
    # viser_main_rrt()
