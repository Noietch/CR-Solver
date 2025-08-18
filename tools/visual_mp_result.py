import jax
import jaxlie
import jax.numpy as jnp
import json
import time
import viser
import numpy as np
import os
import matplotlib.pyplot as plt
from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.robots.tdcr_robot import TDCRRobot
from soul.geom import RobotCollision, WorldCollision

from soul.visualization.visualizer_viser import (
    ViserSoftRobot,
    ViserWorld,
    ViserRenderer,
)
from soul.visualization.visualizer_plot import visualize_cc_model_3d

TICK_LABELSIZE = 25
TICK_PAD = 12


def set_3d_tick_labelsize(
    ax: plt.Axes, size: int = TICK_LABELSIZE, pad: int = TICK_PAD
) -> None:
    ax.xaxis.set_tick_params(labelsize=size, pad=pad, direction="in")
    ax.yaxis.set_tick_params(labelsize=size, pad=pad, direction="in")
    if hasattr(ax, "zaxis"):
        ax.zaxis.set_tick_params(labelsize=size, pad=pad, direction="in")


def viser_main(
    robot_config_path: str,
    world_config_paths: str,
    mp_result_path: str,
    robot_type: str = "tdcr",
):
    # Setup Robot Environment
    config = json.load(open(robot_config_path))
    # num_sections = int(world_config_paths.split(os.sep)[-1].split('_')[-1].split('.')[0])
    num_sections = int(
        mp_result_path.split("_all_trials_trajectories")[0].split("_")[-1]
    )
    config["num_sections"] = num_sections
    if robot_type == "cc":
        robot = CCRobot.from_config(config)
        robot_coll = RobotCollision.from_config(config)
    elif robot_type == "tdcr":
        robot = TDCRRobot.from_config(config)
        robot_coll = RobotCollision.from_config(config)
    world_coll = WorldCollision.from_config(world_config_paths)

    # Load motion planning results
    data = np.load(mp_result_path, allow_pickle=True)

    # Extract trial IDs and data
    trial_ids = sorted(list(set([int(k.split("_")[1]) for k in data.keys()])))
    num_trials = len(trial_ids)
    print(f"Loaded {num_trials} trials with IDs: {trial_ids}")

    # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot, robot_coll, root_node_name="/robot")
    robot_vis.create_robot_visualizations()
    obstacles_vis = ViserWorld(server, world_coll, enable_collision=False)
    obstacles_vis.create_mesh_visualizations()
    renderer = ViserRenderer(server, robot_vis, obstacles_vis)

    # JIT compile forward kinematics
    forward_kinematics = jax.jit(jax.vmap(robot._forward_kinematics))

    # Global state
    current_trial_idx = 0
    solution_pose = None

    # Setup GUI
    with server.gui.add_folder("Trial Navigation"):
        trial_id_text = server.gui.add_text(
            "Current Trial ID",
            initial_value=str(trial_ids[current_trial_idx]),
            disabled=True,
        )
        prev_button = server.gui.add_button("Previous Trial")
        next_button = server.gui.add_button("Next Trial")

    replay_button = server.gui.add_button("Replay Trajectory")
    render_video_button = server.gui.add_button("Render Video")
    render_image_button = server.gui.add_button("Render Image")
    visualize_traj_collisions_bottom = server.gui.add_button(
        "Visualize Trajectory Collisions"
    )
    visualize_mp_button = server.gui.add_button("Save Motion Planning")

    def load_and_display_trial(trial_idx: int):
        global planned_traj
        global planned_paths_constant
        global planned_tip_traj
        trial_id = trial_ids[trial_idx]
        trial_id_text.value = str(trial_id)
        print(f"Loading trial ID: {trial_id} (index: {trial_idx})")

        # Reconstruct configuration trajectory from the new data format
        solution_states_theta = data[f"trial_{trial_id}_solution_states_theta"]
        solution_states_phi = data[f"trial_{trial_id}_solution_states_phi"]
        planned_paths = data[f"trial_{trial_id}_planned_paths"]

        num_steps = solution_states_theta.shape[0]
        base_position = jnp.zeros((num_steps, 3))

        solution_cfg = ConstantCurvatureState(
            base_position=base_position,
            theta=solution_states_theta,
            phi=solution_states_phi,
        )

        # Compute the trajectory using forward kinematics
        solution_pose = forward_kinematics(solution_cfg)
        solution_pose = jax.block_until_ready(solution_pose)

        planned_paths = planned_paths.item()
        planned_paths["base_position"] = planned_paths["base_position"].squeeze(0)
        planned_paths["theta"] = planned_paths["theta"].squeeze(0)
        planned_paths["phi"] = planned_paths["phi"].squeeze(0)

        planned_paths_constant = ConstantCurvatureState.load_from_dict(planned_paths)
        planned_traj = forward_kinematics(planned_paths_constant)
        jax.block_until_ready(planned_traj)

        # Display the first frame of the trajectory
        robot_vis.update_pose(solution_pose[0])

        print(f"Trajectory loaded with {len(planned_traj)} points.")

    def visualize_mp(event: viser.GuiEvent):
        nonlocal current_trial_idx
        global planned_traj
        global planned_tip_traj

        print(f"Saving motion planning visualization of trial {current_trial_idx}...")
        fig = plt.figure(facecolor="white", figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        set_3d_tick_labelsize(ax)
        save_path = os.path.join(
            os.path.dirname(mp_result_path),
            "visualization",
            f"section_{num_sections}_trial_{current_trial_idx}.png",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        num_timesteps = len(planned_traj)
        sample_indices = None
        num_pose = 15
        if sample_indices is None:
            sample_indices = np.linspace(0, num_timesteps - 1, num_pose, dtype=int)
        selected_poses = planned_traj[sample_indices]
        visualize_cc_model_3d(
            pose=selected_poses,
            num_points=10,
            ax=ax,
            world_coll_config=world_config_paths,
        )

        planned_tip_pos = planned_tip_traj[:, :3, 3]
        set_3d_tick_labelsize(ax)
        ax.plot(
            planned_tip_pos[:, 0],
            planned_tip_pos[:, 1],
            planned_tip_pos[:, 2],
            c="#1f77b4",  # blue
            linewidth=3,
            label="Experiment",
        )
        # Swap x and y axes by plotting y as x and x as y
        ax.set_xlim3d(-3, 3)
        ax.set_ylim3d(-3, 3)
        ax.set_zlim3d(-0.5, 3)

        # Set equal aspect ratio (swap x and y dimensions)
        ax.set_box_aspect([6, -6, 3.5])
        ax.patch.set_alpha(0.0)

        # # Swap axis labels
        # ax.set_xlabel("Y")
        # ax.set_ylabel("X")
        # ax.set_zlabel("Z")

        # Swap tick labels (optional, for clarity)
        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        # ax.legend(frameon=False, markerscale=1)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    def on_prev_click(event: viser.GuiEvent):
        nonlocal current_trial_idx
        current_trial_idx = max(0, current_trial_idx - 1)
        load_and_display_trial(current_trial_idx)

    def on_next_click(event: viser.GuiEvent):
        nonlocal current_trial_idx
        current_trial_idx = min(num_trials - 1, current_trial_idx + 1)
        load_and_display_trial(current_trial_idx)

    def on_replay_click(event: viser.GuiEvent):
        global planned_traj
        if planned_traj is None:
            print("No trajectory loaded to replay.")
            return

        print(f"Replaying trajectory for trial ID: {trial_ids[current_trial_idx]}...")
        for i in range(len(planned_traj)):
            robot_vis.update_pose(planned_traj[i])
            time.sleep(1 / 60.0)
        print("Replay finished.")

    def visualize_traj_collisions(event: viser.GuiEvent):
        global planned_traj
        robot_vis.visualize_traj_collisions(robot, planned_paths_constant)

    def render_video_callback(event: viser.GuiEvent):
        global planned_traj
        if planned_traj is None:
            print("No trajectory loaded to render.")
            return

        trial_id = trial_ids[current_trial_idx]
        save_path = f"trajectory_video_trial_{trial_id}.mp4"
        print(f"Rendering video to {save_path}...")
        renderer.render_traj_video(event, planned_traj, save_path=save_path)
        print("Video rendering finished.")

    def render_image_callback(event: viser.GuiEvent):
        global planned_traj
        if planned_traj is None:
            print("No trajectory loaded to render.")
            return

        trial_id = trial_ids[current_trial_idx]
        save_path = f"results/trajectory_image_trial_{trial_id}.png"
        print(f"Rendering image to {save_path}...")
        renderer.render_traj_image(event, planned_traj, save_path=save_path)
        print("Image rendering finished.")

    prev_button.on_click(on_prev_click)
    next_button.on_click(on_next_click)
    replay_button.on_click(on_replay_click)
    render_video_button.on_click(render_video_callback)
    render_image_button.on_click(render_image_callback)
    visualize_traj_collisions_bottom.on_click(visualize_traj_collisions)
    visualize_mp_button.on_click(visualize_mp)

    # Load the first trial initially
    load_and_display_trial(current_trial_idx)

    while True:
        time.sleep(1 / 60.0)


if __name__ == "__main__":

    robot_config_path = "configs/robots/cc_scene_eval_tdcr.json"
    world_config_paths = (
        "configs/maps/mp_scene/obstacles_random_start_init_True_section_3.json"
    )
    mp_result_path = "results/debug_draw/obstacles_random_start_init_True_section_3/all_trials_trajectories/trajopt_opt_False_sections_3_all_trials_trajectories.npz"

    viser_main(
        robot_config_path, world_config_paths, mp_result_path, robot_type="tdcr"
    )  # Change to "cc" for CCRobot
