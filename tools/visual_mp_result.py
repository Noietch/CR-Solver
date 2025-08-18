"""in file trajopt_opt_False_sections_6_all_trials_trajectories.npz
"trial_id": int
"start_states_theta": np.asarray(start_states.theta),
"start_states_phi": np.asarray(start_states.phi),
"end_states_theta": np.asarray(end_states.theta),
"end_states_phi": np.asarray(end_states.phi),
"target_position": np.asarray(target_position),
"target_wxyz": np.asarray(target_wxyz),
"fk_result": np.asarray(fk_result),
"solution_states_theta": np.asarray(solution_states.theta),
"solution_states_phi": np.asarray(solution_states.phi),
"planned_tip_traj": np.asarray(planned_tip_traj.as_matrix()),
格式：
trial_{trial_num}_{key}
"""

# robot_config_path = "configs/robots/cc_scene_eval_tdcr.json"
# world_config_paths = "configs/maps/mp_scene/obstacles_random_section_6.json"
# mp_result = "results/debug_traj_inf_coll/obstacles_random_section_6/all_trials_trajectories/trajopt_opt_False_sections_6_all_trials_trajectories.npz"

robot_config_path = "configs/robots/cc_scene_eval_tdcr.json"
world_config_paths = "configs/maps/mp_scene/obstacles_random_section_6.json"
mp_result_path = "results/debug_traj_inf_coll/obstacles_random_section_6/all_trials_trajectories/trajopt_opt_False_sections_6_all_trials_trajectories.npz"


import jax
import jaxlie
import jax.numpy as jnp
import json
import time
import viser
import numpy as np
import os
from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.robots.tdcr_robot import TDCRRobot
from soul.geom import RobotCollision, WorldCollision

from soul.visualization.visualizer_viser import (
    ViserSoftRobot,
    ViserWorld,
    ViserRenderer,
)


def viser_main(robot_type: str = "tdcr"):
    # Setup Robot Environment
    config = json.load(open(robot_config_path))
    num_sections = int(world_config_paths.split(os.sep)[-1].split('_')[-1].split('.')[0])
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
    trial_ids = sorted(list(set([int(k.split('_')[1]) for k in data.keys()])))
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
    current_traj = None

    # Setup GUI
    with server.gui.add_folder("Trial Navigation"):
        trial_id_text = server.gui.add_text(
            "Current Trial ID", initial_value=str(trial_ids[current_trial_idx]), disabled=True
        )
        prev_button = server.gui.add_button("Previous Trial")
        next_button = server.gui.add_button("Next Trial")
    
    start_handle = server.scene.add_transform_controls(
        "/start",
        scale=0.3,
        position=(0,0,0),
        wxyz=(1, 0, 0, 0),
    )

    target_handle = server.scene.add_transform_controls(
        "/target",
        scale=0.3,
        position=(0,0,0),
        wxyz=(1, 0, 0, 0),
    )
    # target_handle.visible = False # Initially hidden
    # start_handle.visible = False  # Initially hidden

    replay_button = server.gui.add_button("Replay Trajectory")
    render_video_button = server.gui.add_button("Render Video")
    render_image_button = server.gui.add_button("Render Image")
    visualize_traj_collisions_bottom = server.gui.add_button("Visualize Trajectory Collisions")


    def load_and_display_trial(trial_idx: int):
        global planned_traj
        global planned_paths_constant
        trial_id = trial_ids[trial_idx]
        trial_id_text.value = str(trial_id)
        print(f"Loading trial ID: {trial_id} (index: {trial_idx})")

        # Reconstruct configuration trajectory from the new data format
        solution_states_theta = data[f"trial_{trial_id}_solution_states_theta"]
        solution_states_phi = data[f"trial_{trial_id}_solution_states_phi"]
        planned_paths = data[f"trial_{trial_id}_planned_paths"]
        planned_tip_traj = data[f"trial_{trial_id}_planned_tip_traj"]

        # Load and display the target pose
        target_position = data[f"trial_{trial_id}_target_position"]
        target_wxyz = data[f"trial_{trial_id}_target_wxyz"]
        start_position = data[f"trial_{trial_id}_start_position"]
        start_wxyz = data[f"trial_{trial_id}_start_wxyz"]

        target_handle.position = tuple(target_position)
        target_handle.wxyz = tuple(target_wxyz)
        target_handle.visible = True
        start_handle.position = tuple(start_position)
        start_handle.wxyz = tuple(start_wxyz)
        start_handle.visible = True
        
        num_steps = solution_states_theta.shape[0]
        base_position = jnp.zeros((num_steps, 3))

        cfg = ConstantCurvatureState(
            base_position=base_position,
            theta=solution_states_theta,
            phi=solution_states_phi,
        )

        # Compute the trajectory using forward kinematics
        current_traj = forward_kinematics(cfg)
        current_traj = jax.block_until_ready(current_traj)

        planned_paths = planned_paths.item()
        planned_paths['base_position'] = planned_paths['base_position'].squeeze(0)
        planned_paths['theta'] = planned_paths['theta'].squeeze(0)
        planned_paths['phi'] = planned_paths['phi'].squeeze(0)

        planned_paths_constant = ConstantCurvatureState.load_from_dict(planned_paths)
        planned_traj = forward_kinematics(planned_paths_constant)
        jax.block_until_ready(planned_traj)


        # Display the first frame of the trajectory
        robot_vis.update_pose(current_traj[0])

        # planned_tip_traj is already a matrix array from the saved data
        robot_vis.visualize_tip_traj(
            planned_tip_traj, color=np.array([1.0, 0.0, 0.0]), name="planned_tip_traj"
        )

        print(f"Trajectory loaded with {len(planned_traj)} points.")


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

    def visualize_collision(event: viser.GuiEvent):
        global planned_paths_constant
        if planned_paths_constant is None:
            print("No trajectory loaded to visualize.")
            return

        # Visualize the collision between the robot and the environment
        robot_vis.visualize_traj_collisions(robot, planned_paths_constant)

    prev_button.on_click(on_prev_click)
    next_button.on_click(on_next_click)
    replay_button.on_click(on_replay_click)
    render_video_button.on_click(render_video_callback)
    render_image_button.on_click(render_image_callback)
    visualize_traj_collisions_bottom.on_click(visualize_collision)

    # Load the first trial initially
    load_and_display_trial(current_trial_idx)

    while True:
        time.sleep(1 / 60.0)

if __name__ == "__main__":
    viser_main(robot_type="tdcr")  # Change to "cc" for CCRobot