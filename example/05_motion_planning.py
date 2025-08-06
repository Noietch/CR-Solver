import jax
import time
import viser
import numpy as np
from soul.robots.cc_robot import CCRobot
from soul.geom import RobotCollision, WorldCollision
from soul.solver import MotionPlanner, PRMMotionPlanner, RRTMotionPlanner, ParallelPRM
from soul.visualization.visualizer_viser import (
    ViserSoftRobot,
    ViserWorld,
    ViserRenderer,
)

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def viser_main(method: str = "trajopt"):
    # Setup Environment
    robot = CCRobot.from_config("configs/robots/cc.json")
    robot_coll = RobotCollision.from_config("configs/robots/cc.json")
    world_coll = WorldCollision.from_config("configs/maps/mp_scene/mp_demo.json")

    # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot, robot_coll, root_node_name="/robot")
    robot_vis.create_robot_visualizations()
    obstacles_vis = ViserWorld(server, world_coll, enable_collision=False)
    obstacles_vis.create_mesh_visualizations()
    renderer = ViserRenderer(server, robot_vis, obstacles_vis)

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
    render_video_button = server.gui.add_button("Render Video", disabled=False)
    render_image_button = server.gui.add_button("Render Image", disabled=False)

    # Set up trajopt parameters
    timesteps = 100
    traj_solver = MotionPlanner(robot, robot_coll, timesteps)
    start_end_interpolate_jit = jax.jit(traj_solver.start_end_interpolate)

    # init motion planner
    trajopt_solver = jax.jit(traj_solver.optimize)
    rrt_traj_solver = RRTMotionPlanner(robot, robot_coll, timesteps)
    prm_traj_solver = PRMMotionPlanner(robot, robot_coll, timesteps)
    test_prm_traj_solver = ParallelPRM(robot, robot_coll)
    traj = None

    def plan_callback(args):
        print("Start planning....")
        global traj

        if method == "trajopt":
            cfg = start_end_interpolate_jit(
                start_handle.position,
                start_handle.wxyz,
                end_handle.position,
                end_handle.wxyz,
                world_coll.collision_geoms,
            )
            cfg = trajopt_solver(cfg, world_coll.collision_geoms)

        elif method == "prm":
            results = prm_traj_solver._ik_solver_best(
                start_handle.wxyz,
                start_handle.position,
                end_handle.wxyz,
                end_handle.position,
                world_coll.collision_geoms,
            )
            cfg = prm_traj_solver.find_path(
                results[0],
                results[1],
                world_coll.collision_geoms,
            )
            if cfg is None:
                print("No path found")
                return

        elif method == "rrt":
            results = rrt_traj_solver._ik_solver_best(
                start_handle.wxyz,
                start_handle.position,
                end_handle.wxyz,
                end_handle.position,
                world_coll.collision_geoms,
            )
            cfg = rrt_traj_solver.find_path(
                results[0],
                results[1],
                world_coll.collision_geoms,
            )
            if cfg is None:
                print("No path found")
                return

        elif method == "test":
            results = rrt_traj_solver._ik_solver_best(
                start_handle.wxyz,
                start_handle.position,
                end_handle.wxyz,
                end_handle.position,
                world_coll.collision_geoms,
            )
            # test_prm_traj_solver.build_roadmap(
            #     5000,
            #     world_coll.collision_geoms,
            # )
            # test_prm_traj_solver.save_roadmap("roadmap.pkl")
            test_prm_traj_solver.load_roadmap("roadmap.pkl")
            cfg = test_prm_traj_solver.find_path(
                results[0],
                results[1],
                world_coll.collision_geoms,
            )
            if cfg is None:
                print("No path found")
                return

        traj = robot.forward_kinematics(cfg)
        print("Finish planning....")
        # robot_vis.visualize_traj_collisions(robot, cfg)
        # for i in range(timesteps):
        #     time.sleep(1/60.0)
        #     robot_vis.update_pose(traj[i])

    def replay_callback(args):
        print("Start replaying....")
        global traj
        if traj is None:
            return
        for i in range(len(traj)):
            robot_vis.update_pose(traj[i])
        print("Finish replaying....")

    def render_video_callback(event: viser.GuiEvent):
        global traj
        if traj is None:
            return
        renderer.render_traj_video(event, traj, save_path="trajectory_video.mp4")

    def render_image_callback(event: viser.GuiEvent):
        global traj
        if traj is None:
            return
        renderer.render_traj_image(event, traj, save_path="trajectory_image.png")

    def on_handle_update(handle: viser.TransformControlsHandle):
        """Update GUI when handles are moved."""
        start_pos_text.value = str(np.round(start_handle.position, 2))
        start_wxyz_text.value = str(np.round(start_handle.wxyz, 2))
        end_pos_text.value = str(np.round(end_handle.position, 2))
        end_wxyz_text.value = str(np.round(end_handle.wxyz, 2))

    plan_button.on_click(plan_callback)
    replay_button.on_click(replay_callback)
    render_video_button.on_click(render_video_callback)
    render_image_button.on_click(render_image_callback)
    start_handle.on_update(on_handle_update)
    end_handle.on_update(on_handle_update)
    on_handle_update(start_handle)

    while True:
        time.sleep(1 / 60.0)


if __name__ == "__main__":
    viser_main(method="test")
