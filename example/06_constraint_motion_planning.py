import time

import jax
import jaxlie
import numpy as np
import viser

from soul.geom import RobotCollision, WorldCollision
from soul.robots.cc_robot import CCRobot
from soul.solver.traj_optimizer import TrajOptimizer, TrajOptimizerOptions
from soul.visualization.visualizer_viser import (
    ViserRenderer,
    ViserSoftRobot,
    ViserWorld,
)

DISABLE_JIT = False

if DISABLE_JIT:
    import os

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def get_linear_traj(
    start_position: np.ndarray,
    start_wxyz: np.ndarray,
    end_position: np.ndarray,
    end_wxyz: np.ndarray,
    timesteps: int,
) -> jaxlie.SE3:
    start_position = np.array(start_position)
    start_wxyz = np.array(start_wxyz)
    end_position = np.array(end_position)
    end_wxyz = np.array(end_wxyz)
    traj_positions = np.linspace(start_position, end_position, timesteps)
    traj_wxyz = np.linspace(start_wxyz, end_wxyz, timesteps)
    traj = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=traj_wxyz), translation=traj_positions
    )
    return traj


def viser_main():
    # Setup Environment
    robot = CCRobot.from_config("configs/robots/cc.json")
    robot_coll = RobotCollision.from_config("configs/robots/cc.json")
    world_coll = WorldCollision.from_config(
        "configs/maps/ik_maps/obstacles_test.json"
    )

    # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(
        server, robot, robot_coll, root_node_name="/robot"
    )
    robot_vis.create_robot_visualizations()
    obstacles_vis = ViserWorld(server, world_coll)
    obstacles_vis.create_mesh_visualizations()
    render = ViserRenderer(server, robot_vis, obstacles_vis)

    # Setup GUI
    start_handle = server.scene.add_transform_controls(
        "/start",
        scale=0.3,
        position=(0.0, 0.0, 3.0),
        wxyz=(1, 0, 0, 0),
    )
    end_handle = server.scene.add_transform_controls(
        "/end", scale=0.3, position=(0.0, 0.0, 5.0), wxyz=(1, 0, 0, 0)
    )
    plan_button = server.gui.add_button("Plan", disabled=False)
    replay_button = server.gui.add_button("Replay", disabled=False)

    # Add GUI for showing handle poses
    with server.gui.add_folder("Handles Tfs"):
        start_pos_text = server.gui.add_text(
            "Start Pos",
            initial_value=str(np.round(start_handle.position, 2)),
            disabled=True,
        )
        start_wxyz_text = server.gui.add_text(
            "Start wxyz",
            initial_value=str(np.round(start_handle.wxyz, 2)),
            disabled=True,
        )
        end_pos_text = server.gui.add_text(
            "End Pos",
            initial_value=str(np.round(end_handle.position, 2)),
            disabled=True,
        )
        end_wxyz_text = server.gui.add_text(
            "End wxyz",
            initial_value=str(np.round(end_handle.wxyz, 2)),
            disabled=True
        )

    # Set up reference trajectory
    timesteps = 100

    def update_reference_traj(args):
        reference_traj = get_linear_traj(
            start_handle.position,
            start_handle.wxyz,
            end_handle.position,
            end_handle.wxyz,
            timesteps,
        )
        robot_vis.visualize_tip_traj(
            reference_traj,
            color=np.array([1.0, 0.0, 0.0]),
            name="reference_traj"
        )
        return reference_traj

    update_reference_traj(None)

    def on_handle_update(handle: viser.TransformControlsHandle):
        """Update GUI and reference trajectory when handles are moved."""
        start_pos_text.value = str(np.round(start_handle.position, 2))
        start_wxyz_text.value = str(np.round(start_handle.wxyz, 2))
        end_pos_text.value = str(np.round(end_handle.position, 2))
        end_wxyz_text.value = str(np.round(end_handle.wxyz, 2))
        update_reference_traj(handle)

    # Set up trajopt parameters
    options = TrajOptimizerOptions(
        collision_weight=0.0,
        smoothness_weight=50,
    )
    traj_solver = TrajOptimizer(robot, robot_coll, timesteps, options)
    traj_follow_jit = jax.jit(traj_solver.optimize_tip_traj_follow)

    traj = None

    def plan_callback(args):
        print("Start planning....")
        global traj

        # Update reference trajectory with current handle positions
        current_reference_traj = update_reference_traj(args)

        cfg = traj_follow_jit(
            current_reference_traj,
            world_coll.collision_geoms,
        )
        traj = robot.forward_kinematics(cfg)
        print("Finish planning....")
        robot_vis.visualize_tip_traj(
            traj, color=np.array([0.0, 0.0, 1.0]), name="planned_traj"
        )
        for i in range(timesteps):
            time.sleep(0.01)
            robot_vis.update_pose(traj[i])

    def replay_callback(event):
        if traj is None:
            return
        render.render_traj_image(event, traj, skip_frames=20, save_path=None)
        # for i in range(timesteps):
        #     time.sleep(0.01)
        #     robot_vis.update_pose(traj[i])

    plan_button.on_click(plan_callback)
    replay_button.on_click(replay_callback)
    start_handle.on_update(on_handle_update)
    end_handle.on_update(on_handle_update)

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    viser_main()
