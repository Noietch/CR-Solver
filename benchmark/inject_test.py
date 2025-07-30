import jax.numpy as jnp
import numpy as np
import json
import os
import jax.tree_util

from soul.robots.cc_robot import CCRobot
from soul.solver import MotionPlanner
from soul.geom import RobotCollision, WorldCollision

# Path to the original sampled data file
INPUT_SAMPLED_FILE = "results/inject/sampled_states/sections_3_eval_1.npz"

# Path for the new file with the injected pose
OUTPUT_MODIFIED_FILE = "results/inject/sampled_states/sections_3_eval_1_injected.npz"

# Robot and world configuration files
ROBOT_CONFIG_PATH = "configs/robots/cc_scene_eval.json"
WORLD_CONFIG_PATH = "configs/maps/mp_scene/obstacles_13.pick_from_shelf.json"
NUM_SECTIONS = 3  # The number of sections must match the robot config in the sampled file

# The specific start and end poses you want to inject at the first position
INJECTED_START_POSE = jnp.array([-0.3, -1.26, 2.51])
INJECTED_START_WXYZ = jnp.array([1, 0, 0, 0])
INJECTED_TARGET_POSE = jnp.array([-0.4, 1.45, 0.89])
INJECTED_TARGET_WXYZ = jnp.array([1, 0, 0, 0])

def inject_pose_into_sampled_file():
    """
    Loads a sampled states file, replaces the first entry with a configuration
    derived from a specific pose via Inverse Kinematics (IK), and saves it
    to a new file.
    """
    print("--- Starting Pose Injection Script ---")

    # Check if the input file exists
    if not os.path.exists(INPUT_SAMPLED_FILE):
        print(f"[Error] Input file not found at: {INPUT_SAMPLED_FILE}")
        return

    print(f"Loading original sampled data from: {INPUT_SAMPLED_FILE}")
    data = np.load(INPUT_SAMPLED_FILE)
    start_theta = data["start_theta"]
    start_phi = data["start_phi"]
    end_theta = data["end_theta"]
    end_phi = data["end_phi"]
    print(f"Loaded {start_theta.shape[0]} samples.")

    # Set up robot and solver to perform IK
    print("Setting up robot and solver for IK...")
    config = json.load(open(ROBOT_CONFIG_PATH))
    config["num_sections"] = NUM_SECTIONS
    robot = CCRobot.from_config(config)
    robot_coll = RobotCollision.from_config(config)
    world_coll = WorldCollision.from_config(WORLD_CONFIG_PATH)
    world_geom = world_coll.collision_geoms_no_ground[-1]

    timesteps = 100
    traj_solver = MotionPlanner(robot, robot_coll, timesteps)
    start_end_interpolate_jit = jax.jit(traj_solver.start_end_interpolate)
    optimize_jit = jax.jit(traj_solver.optimize)

    print("Solving IK for the injectable start and end poses...")
    cfg_pair = start_end_interpolate_jit(
        INJECTED_START_POSE,
        INJECTED_START_WXYZ,
        INJECTED_TARGET_POSE,
        INJECTED_TARGET_WXYZ,
        [world_geom],
    )
    cfg = optimize_jit(cfg_pair, [world_geom])

    start_state_ik = jax.tree_util.tree_map(lambda x: x[0], cfg)
    end_state_ik = jax.tree_util.tree_map(lambda x: x[-1], cfg)
    
    print("IK solution found successfully.")
    print(f"  - Start IK Theta: {start_state_ik.theta}")
    print(f"  - End IK Theta:   {end_state_ik.theta}")


    # Replace the first entry in the loaded data arrays
    print("Replacing the first sample in the dataset...")
    start_theta[0, :] = np.asarray(start_state_ik.theta.flatten())
    start_phi[0, :] = np.asarray(start_state_ik.phi.flatten())
    end_theta[0, :] = np.asarray(end_state_ik.theta.flatten())
    end_phi[0, :] = np.asarray(end_state_ik.phi.flatten())

    # Save the modified data to a new .npz file
    print(f"Saving modified data to: {OUTPUT_MODIFIED_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_MODIFIED_FILE), exist_ok=True)
    np.savez(
        OUTPUT_MODIFIED_FILE,
        start_theta=start_theta,
        start_phi=start_phi,
        end_theta=end_theta,
        end_phi=end_phi,
    )

    print("--- Injection complete! ---")
    print(f"The new file is ready at '{OUTPUT_MODIFIED_FILE}'. "
          f"You can now use this file for evaluation.")


if __name__ == "__main__":
    inject_pose_into_sampled_file()
