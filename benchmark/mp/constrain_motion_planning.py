import os

import jax

# Initialize JAX persistent compilation cache
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir"
)
from jax.experimental.compilation_cache import compilation_cache as cc

cc.set_cache_dir("/tmp/jax_cache")
import csv
import time
from typing import Sequence

import jax.numpy as jnp
import jaxlie
import matplotlib.pyplot as plt
import numpy as np
from benchmark.mp.mp_plot import (
    plot_constrain_motion_planning,
    plot_error,
    plot_tendon_length,
    visualize_constrain_motion_planning,
)

from soul.geom import CollGeom, RobotCollision, WorldCollision
from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.robots.cc_robot_extend import CCRobot as CCRobotExtend
from soul.robots.tdcr_robot import TDCRRobot
from soul.solver.traj_optimizer import TrajOptimizer

DISABLE_JIT = False
if DISABLE_JIT:
    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)

LETTER_HEIGHT = 2.5
LETTER_WIDTH = 1.5


@jax.jit
def is_trajectory_in_collision(
    trajectory: ConstantCurvatureState,
    robot: CCRobot,
    robot_coll: RobotCollision,
    world_geom: Sequence[CollGeom],
) -> bool:
    """Checks if any state in a trajectory is in collision."""
    trajectory = jax.tree_util.tree_map(
        lambda x: jnp.expand_dims(x, axis=0), trajectory
    )
    if trajectory is None or trajectory.theta.shape[0] == 0:
        return True  # No path found is a collision/failure

    # Vmap the single-state check over the trajectory timesteps
    in_collision_mask = jax.vmap(
        is_state_in_collision, in_axes=(0, None, None, None)
    )(trajectory, robot, robot_coll, world_geom)
    return jnp.any(in_collision_mask)


@jax.jit
def is_state_in_collision(
    state: ConstantCurvatureState,
    robot: CCRobot,
    robot_coll: RobotCollision,
    world_geom: Sequence[CollGeom],
) -> bool:
    """
    Check if the robot is in collision with obstacles or itself.

    A collision is defined as any distance less than 0 (i.e., penetration)
    between the robot and any world geometry, or between parts of the robot
    itself.
    """

    def check_single_geom(geom: CollGeom) -> bool:
        world_dist = robot_coll.compute_world_collision_distance(
            robot, state, geom
        )
        return jnp.any(world_dist < 0.0)

    collision_results = jnp.array([check_single_geom(g) for g in world_geom])

    return jnp.any(collision_results)


def create_I():
    """Generates path points for the letter 'I'."""
    return [(LETTER_WIDTH / 2, 0), (LETTER_WIDTH / 2, LETTER_HEIGHT)]


def create_C():
    """Generates path points for a curved letter 'C'."""
    return [(
        LETTER_WIDTH / 2 + (LETTER_WIDTH / 2) * np.cos(t),
        LETTER_HEIGHT / 2 + (LETTER_HEIGHT / 2) * np.sin(t),
    ) for t in np.linspace(0.4 * np.pi, 1.6 * np.pi, 20)]


def create_R():
    """Generates path points for curved letter 'R'."""
    stem = [(0, 0), (0, LETTER_HEIGHT)]
    curve_center_y = LETTER_HEIGHT * 0.75
    curve_radius_y = LETTER_HEIGHT * 0.25
    curve_radius_x = LETTER_WIDTH
    curve = [(
        curve_radius_x * np.cos(t),
        curve_center_y + curve_radius_y * np.sin(t)
    ) for t in np.linspace(np.pi / 2, -np.pi / 3, 15)]
    leg_start_point = (curve[-1][0], curve[-1][1])
    leg_end_point = (LETTER_WIDTH, 0)
    leg = [leg_start_point, leg_end_point]
    return stem + [None] + curve + leg


def create_A():
    """Generates path points for the letter 'A'."""
    return [
        (0, 0),
        (LETTER_WIDTH / 2, LETTER_HEIGHT),
        (LETTER_WIDTH, 0),
        None,
        # (LETTER_WIDTH * 0.3, LETTER_HEIGHT / 3),
        # (LETTER_WIDTH * 0.8, LETTER_HEIGHT / 3),
    ]


def get_icra_traj(
    start_position: np.ndarray,
    start_wxyz: np.ndarray,
    end_position: np.ndarray,
    end_wxyz: np.ndarray,  # Unused but kept for consistent signature
    timesteps: int,
    word_funcs: dict = None,
    word_str: str = "ICRA",
) -> jaxlie.SE3:
    """
    Generates a 3D word trajectory starting at the start_handle's pose.

    The width is controlled by the x-difference between handles.
    The height is controlled by the y-difference between handles.
    """
    # Ensure array types for jaxlie
    start_position = np.array(start_position)
    start_wxyz = np.array(start_wxyz)
    end_position = np.array(end_position)
    end_wxyz = np.array(end_wxyz)
    # Default to only letter A if not provided
    if word_funcs is None:
        word_funcs = {"A": create_A}

    letter_spacing = LETTER_WIDTH * 1.8

    # 1. Generate base 2D letter points in a local frame
    all_points_2d = []
    current_x_offset = 0
    for char in word_str:
        if char in word_funcs:
            points = word_funcs[char]()
            if all_points_2d and points:
                all_points_2d.append(None)
            for p in points:
                if p is None:
                    all_points_2d.append(None)
                else:
                    all_points_2d.append((p[0] + current_x_offset, p[1]))
            if any(p is not None for p in points):
                current_x_offset += letter_spacing

    path_2d_array = np.array([p for p in all_points_2d if p is not None])

    if path_2d_array.shape[0] < 2:
        return get_linear_traj(
            start_position, start_wxyz, start_position, start_wxyz, timesteps
        )

    # 2. Determine target dimensions from handle positions
    handle_diff = np.array(end_position) - np.array(start_position)
    target_width = abs(handle_diff[0])
    target_height = abs(handle_diff[1])

    # Use small default values to prevent collapsing the shape
    if target_width < 0.1:
        target_width = 0.1
    if target_height < 0.1:
        target_height = 0.1

    # 3. Calculate and apply non-uniform scaling
    natural_width = path_2d_array[:, 0].max() - path_2d_array[:, 0].min()
    natural_height = path_2d_array[:, 1].max() - path_2d_array[:, 1].min()

    scale_x = target_width / natural_width if natural_width > 1e-6 else 1.0
    scale_y = target_height / natural_height if natural_height > 1e-6 else 1.0

    # Apply scaling to a copy of the points
    scaled_path_2d = np.copy(path_2d_array)
    # Shift to origin before scaling to maintain shape relative to the start
    min_coords = path_2d_array.min(axis=0)
    scaled_path_2d -= min_coords
    scaled_path_2d[:, 0] *= scale_x
    scaled_path_2d[:, 1] *= scale_y

    # 4. Lift to 3D and align the path's start to the local origin
    local_points_3d = np.hstack([
        scaled_path_2d, np.zeros((scaled_path_2d.shape[0], 1))
    ])

    # 5. Define the world transformation from the start handle's pose
    start_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=start_wxyz), translation=start_position
    )

    # 6. Apply the transformation to map local points to world space
    world_points = start_pose.apply(local_points_3d)

    # 7. Resample the final 3D path for a smooth trajectory
    distances = np.cumsum(
        np.linalg.norm(np.diff(world_points, axis=0), axis=1)
    )
    distances = np.insert(distances, 0, 0)

    new_distances = np.linspace(0, distances[-1], timesteps)
    interp_x = np.interp(new_distances, distances, world_points[:, 0])
    interp_y = np.interp(new_distances, distances, world_points[:, 1])
    interp_z = np.interp(new_distances, distances, world_points[:, 2])
    traj_positions = np.column_stack([interp_x, interp_y, interp_z])

    # 8. Use the start handle's orientation for the entire trajectory
    traj_wxyz = np.tile(np.array(start_wxyz), (timesteps, 1))

    # 9. Create the final SE3 trajectory object
    traj = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=traj_wxyz), translation=traj_positions
    )
    return traj


@jax.jit
def calculate_traj_metrics(
    reference_traj: jaxlie.SE3,
    planned_traj: jaxlie.SE3,
) -> dict:
    """
    Calculates trajectory following metrics.

    Args:
        reference_traj: The target trajectory.
        planned_traj: The executed trajectory.

    Returns:
        A dictionary with per-timestep errors plus summary stats.
    """
    # Position error
    ref_pos = reference_traj.translation()
    plan_pos = planned_traj.translation()
    position_errors = jnp.linalg.norm(plan_pos - ref_pos, axis=-1)

    # Rotation error
    ref_rot = reference_traj.rotation()
    plan_rot = planned_traj.rotation()
    rotation_errors = jnp.linalg.norm((ref_rot.inverse() @ plan_rot).log(),
                                      axis=-1)

    metrics = {
        "position_errors": position_errors,
        "rotation_errors": rotation_errors,
        "pos_error_mean": jnp.mean(position_errors),
        "pos_error_std": jnp.std(position_errors),
        "rot_error_mean": jnp.mean(rotation_errors),
        "rot_error_std": jnp.std(rotation_errors),
    }
    return metrics


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


def create_square():
    """Generates path points for a square."""
    side_length = LETTER_WIDTH  # Use LETTER_WIDTH as the base side length
    return [
        (0, 0),
        (side_length, 0),
        (side_length, side_length),
        (0, side_length),
        (0, 0),  # Close the loop
    ]


def get_square_traj(
    start_position: np.ndarray,
    start_wxyz: np.ndarray,
    end_position: np.ndarray,
    end_wxyz: np.ndarray,  # Unused but kept for consistent signature
    timesteps: int,
) -> jaxlie.SE3:
    """Generates a 3D square trajectory based on handle positions."""
    start_position = np.array(start_position)
    start_wxyz = np.array(start_wxyz)
    end_position = np.array(end_position)
    end_wxyz = np.array(end_wxyz)
    points_2d = np.array(create_square())
    handle_diff = np.array(end_position) - np.array(start_position)
    target_side_length = max(abs(handle_diff[0]), 0.1)

    natural_side_length = points_2d[:, 0].max() - points_2d[:, 0].min()
    scale = (
        target_side_length
        / natural_side_length if natural_side_length > 1e-6 else 1.0
    )
    scaled_path_2d = points_2d * scale

    local_points_3d = np.hstack([
        scaled_path_2d, np.zeros((scaled_path_2d.shape[0], 1))
    ])
    start_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=start_wxyz), translation=start_position
    )
    world_points = start_pose.apply(local_points_3d)

    distances = np.cumsum(
        np.linalg.norm(np.diff(world_points, axis=0), axis=1)
    )
    distances = np.insert(distances, 0, 0)
    new_distances = np.linspace(0, distances[-1], timesteps)

    interp_coords = [
        np.interp(new_distances, distances, world_points[:, i])
        for i in range(3)
    ]
    traj_positions = np.column_stack(interp_coords)
    traj_wxyz = np.tile(np.array(start_wxyz), (timesteps, 1))

    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=traj_wxyz), translation=traj_positions
    )


def get_sine_traj(
    start_position: np.ndarray,
    start_wxyz: np.ndarray,
    end_position: np.ndarray,
    end_wxyz: np.ndarray,
    timesteps: int,
):
    start_position = np.array(start_position)
    start_wxyz = np.array(start_wxyz)
    end_position = np.array(end_position)
    end_wxyz = np.array(end_wxyz)

    y_coords = np.linspace(start_position[1], end_position[1], timesteps)
    z_coords = np.linspace(start_position[2], end_position[2], timesteps)
    x_base = np.linspace(start_position[0], end_position[0], timesteps)

    t = np.linspace(0, 2 * np.pi, timesteps)
    x_amplitude = 0.3
    x_sine = x_base + x_amplitude * np.sin(t)

    traj_positions = np.column_stack([x_sine, y_coords, z_coords])
    traj_wxyz = np.linspace(start_wxyz, end_wxyz, timesteps)

    traj = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=traj_wxyz), translation=traj_positions
    )
    return traj


def save_ref_traj_to_csv(reference_traj: jaxlie.SE3, file_path: str):
    """Saves the reference trajectory to a CSV file.

    Args:
        reference_traj: The SE3 trajectory to save.
        file_path: The path to the CSV file.
    """
    translations = np.asarray(reference_traj.translation())
    # wxyz is equivalent to qw, qx, qy, qz
    quaternions_wxyz = np.asarray(reference_traj.rotation().wxyz)

    data_to_save = np.hstack([translations, quaternions_wxyz])

    header = ["x", "y", "z", "qw", "qx", "qy", "qz"]

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_to_save)
    print(f"Saved reference trajectory to {file_path}")


def run_case(
    robot: CCRobot,
    traj_follow_jit,
    world_coll_config_path: str,
    traj_type: str,
    letters: str,
    start_handle_position: tuple,
    start_handle_wxyz: tuple,
    end_handle_position: tuple,
    end_handle_wxyz: tuple,
    timesteps: int = 150,
):
    # Setup world collision per case
    world_coll = WorldCollision.from_config(world_coll_config_path)
    world_geom_list = world_coll.collision_geoms_no_ground

    vmapped_is_trajectory_in_collision = jax.vmap(
        is_trajectory_in_collision, in_axes=(0, None, None, None)
    )

    # Select reference trajectory
    if traj_type == "linear":
        ref_func = get_linear_traj
        ref_traj = ref_func(
            start_handle_position,
            start_handle_wxyz,
            end_handle_position,
            end_handle_wxyz,
            timesteps,
        )
    elif traj_type == "square":
        ref_func = get_square_traj
        ref_traj = ref_func(
            start_handle_position,
            start_handle_wxyz,
            end_handle_position,
            end_handle_wxyz,
            timesteps,
        )
    elif traj_type == "sine":
        ref_func = get_sine_traj
        ref_traj = ref_func(
            start_handle_position,
            start_handle_wxyz,
            end_handle_position,
            end_handle_wxyz,
            timesteps,
        )
    else:  # icra word/letters
        ref_func = get_icra_traj
        all_letter_funcs = {
            "I": create_I,
            "C": create_C,
            "R": create_R,
            "A": create_A
        }
        selected_funcs = {
            ch: all_letter_funcs[ch]
            for ch in letters
            if ch in all_letter_funcs
        }
        if not selected_funcs:
            raise ValueError(
                "No valid letters selected for ICRA trajectory. "
                "Choose from I, C, R, A."
            )
        ref_traj = ref_func(
            start_handle_position,
            start_handle_wxyz,
            end_handle_position,
            end_handle_wxyz,
            timesteps,
            word_funcs=selected_funcs,
            word_str=letters,
        )

    # Plan trajectory
    print("Start planning....")
    # Warmup: run once to trigger JIT compilation before timing
    cfg = traj_follow_jit(ref_traj, [world_coll.obstacles])
    is_in_collision = vmapped_is_trajectory_in_collision(
        cfg, robot, robot_coll, world_geom_list
    )
    jax.block_until_ready(is_in_collision)

    start_time = time.time()
    cfg = traj_follow_jit(ref_traj, [world_coll.obstacles])
    is_in_collision = vmapped_is_trajectory_in_collision(
        cfg, robot, robot_coll, world_geom_list
    )
    jax.block_until_ready(is_in_collision)
    end_time = time.time()
    if is_in_collision.any():
        raise Exception("Planned trajectory is in collision!")
    planning_time = end_time - start_time
    print(f"Planning finished in {planning_time:.4f} seconds.")

    # Compute TDCR tendon lengths vs time if applicable
    tendon_lengths_series = None
    if isinstance(robot, TDCRRobot):
        calc_tendon_lengths_batch = jax.jit(
            jax.vmap(robot.calculate_tendon_lengths)
        )
        tendon_lengths_series = np.asarray(calc_tendon_lengths_batch(cfg))

    # FK and metrics
    global_traj = robot.forward_kinematics(cfg)
    planned_tip_traj = jaxlie.SE3.from_matrix(global_traj[:, -1, :, :])

    calculate_traj_metrics_jit = jax.jit(calculate_traj_metrics)
    metrics = calculate_traj_metrics_jit(ref_traj, planned_tip_traj)
    print("--- Trajectory Metrics ---")
    print(f"Position Error (mean): {metrics['pos_error_mean']:.4f}m")
    print(f"Position Error (std):  {metrics['pos_error_std']:.4f}m")
    print(f"Rotation Error (mean): {metrics['rot_error_mean']:.4f}rad")
    print(f"Rotation Error (std):  {metrics['rot_error_std']:.4f}rad")
    print(f"Planning Time: {planning_time:.4f}s")

    # Save results (unique per case)
    case_suffix = letters if traj_type == "icra" else traj_type
    save_dir = os.path.join(
        "results", "traj_following", f"{ref_func.__name__}_{case_suffix}"
    )
    os.makedirs(save_dir, exist_ok=True)

    ref_csv_filename = f"{ref_func.__name__}_reference.csv"
    ref_csv_save_path = os.path.join(save_dir, ref_csv_filename)
    save_ref_traj_to_csv(ref_traj, ref_csv_save_path)

    # Save TDCR tendon lengths series to CSV
    if tendon_lengths_series is not None:
        csv_path = os.path.join(save_dir, "tendon_lengths.csv")
        time_series = np.arange(
            tendon_lengths_series.shape[0]
        )  # timestep index
        header = ",".join(
            ["timestep"]
            + [f"tendon_{i+1}" for i in range(tendon_lengths_series.shape[1])]
        )
        data = np.column_stack([time_series, tendon_lengths_series])
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
        print(f"Saved TDCR tendon length time series to {csv_path}")

    filename = f"{ref_func.__name__}.npz"
    save_path = os.path.join(save_dir, filename)

    obstacles_for_saving = world_coll.obstacles

    if isinstance(robot, CCRobotExtend):
        np.savez(
            save_path,
            solution_states_theta=np.asarray(cfg.theta),
            solution_states_phi=np.asarray(cfg.phi),
            solution_states_length=np.asarray(cfg.length),
            obstacles=obstacles_for_saving,
            fk_result=np.asarray(global_traj),
            target_position=np.asarray(ref_traj.translation()),
            target_wxyz=np.asarray(ref_traj.rotation().wxyz),
            planned_tip_traj=np.asarray(planned_tip_traj.as_matrix()),
            planning_time=planning_time,
            **{
                k: np.array(v)
                for k, v in metrics.items()
            },
        )
    else:
        np.savez(
            save_path,
            solution_states_theta=np.asarray(cfg.theta),
            solution_states_phi=np.asarray(cfg.phi),
            obstacles=obstacles_for_saving,
            fk_result=np.asarray(global_traj),
            target_position=np.asarray(ref_traj.translation()),
            target_wxyz=np.asarray(ref_traj.rotation().wxyz),
            planned_tip_traj=np.asarray(planned_tip_traj.as_matrix()),
            planning_time=planning_time,
            **{
                k: np.array(v)
                for k, v in metrics.items()
            },
        )
    print(f"Saved trajectory data to {save_path}")

    # Plot and save the figure
    fig = plt.figure(facecolor="white", figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    visualize_constrain_motion_planning(
        save_path=save_path,
        world_config_path=world_coll_config_path,
        ax=ax,
    )

    plt.tight_layout()
    fig_save_path = os.path.join(save_dir, "motion_planning_examples.png")
    plt.savefig(fig_save_path)
    print(f"Saved plot to {fig_save_path}")
    plt.close()


if __name__ == "__main__":
    # Select robot type here without CLI: set to "cc" or "tdcr"
    robot_type = "tdcr"  # change to "cc" to use constant-curvature robot
    timesteps = 150

    # Choose config based on robot type and construct robot + collision
    if robot_type == "cc":
        robot_config = "configs/robots/cc_con_eval.json"
        robot = CCRobot.from_config(robot_config)
        robot_coll = RobotCollision.from_config(robot_config)
    elif robot_type == "tdcr":
        robot_config = "configs/robots/cc_con_tdcr.json"
        robot = TDCRRobot.from_config(robot_config)
        robot_coll = RobotCollision.from_config(robot_config)

    # Create JIT-compiled planner once for all cases
    traj_solver = TrajOptimizer(robot, robot_coll, timesteps)
    traj_follow_jit = jax.jit(traj_solver.optimize_tip_traj_follow)

    cases = [
        # {
        #     "name": "square",
        #     "traj": "square",
        #     "obstacle":
        #         "configs/maps/constrain_motion_planning/"
        #         "obstacles_con_square.json",
        #     "start_pos": (-0.05, -2.3, 0.96),
        #     "start_wxyz": (0.83, 0.54, 0.06, 0.1),
        #     "end_pos": (0.7, -1.45, 2.75),
        #     "end_wxyz": (1.0, 0, 0, 0),
        #     "letters": "",
        # },
        # {
        #     "name": "sine",
        #     "traj": "sine",
        #     "obstacle":
        #         "configs/maps/constrain_motion_planning/"
        #         "obstacles_con_sine.json",
        #     "start_pos": (0.5, -2.11, 0.89),
        #     "start_wxyz": (1.0, 0.0, 0.0, 0.0),
        #     "end_pos": (0.51, -2.01, 1.49),
        #     "end_wxyz": (1.0, 0.0, 0.0, 0.0),
        #     "letters": "",
        # },
        {
            "name": "sine",
            "traj": "sine",
            "obstacle": "configs/maps/constrain_motion_planning/"
            "obstacles_con_sine.json",
            "start_pos": (0.48, 1.35, 1.54),
            "start_wxyz": (0.71, 0.71, 0.0, 0.0),
            "end_pos": (0.37, 0.86, 2.02),
            "end_wxyz": (0.71, 0.71, 0.0, 0.0),
            "letters": "",
        },
        {
            "name": "A",
            "traj": "icra",
            "obstacle": "configs/maps/constrain_motion_planning/"
            "obstacles_con_A.json",
            "start_pos": (-0.01, -2.51, 0.76),
            "start_wxyz": (0.82, 0.47, 0.25, 0.2),
            "end_pos": (1.23, -1.25, 2.75),
            "end_wxyz": (1.0, 0.0, 0.0, 0.0),
            "letters": "A",
        },
        {
            "name": "R",
            "traj": "icra",
            "obstacle": "configs/maps/constrain_motion_planning/"
            "obstacles_con_R.json",
            "start_pos": (-0.11, -2.36, 0.87),
            "start_wxyz": (0.85, 0.49, 0.15, 0.14),
            "end_pos": (0.79, -1.26, 2.75),
            "end_wxyz": (1.0, 0.0, 0.0, 0.0),
            "letters": "R",
        },
        {
            "name": "C",
            "traj": "icra",
            "obstacle": "configs/maps/constrain_motion_planning/"
            "obstacles_con_C.json",
            "start_pos": (-0.01, -2.46, 0.9),
            "start_wxyz": (0.83, 0.54, 0.06, 0.1),
            "end_pos": (0.75, -1.31, 2.75),
            "end_wxyz": (1.0, 0.0, 0.0, 0.0),
            "letters": "C",
        },
        {
            "name": "I",
            "traj": "icra",
            "obstacle": "configs/maps/constrain_motion_planning/"
            "obstacles_con_I.json",
            "start_pos": (-0.01, -2.45, 0.91),
            "start_wxyz": (0.83, 0.54, 0.06, 0.1),
            "end_pos": (0.75, -1.37, 2.75),
            "end_wxyz": (1.0, 0.0, 0.0, 0.0),
            "letters": "I",
        },
    ]

    for case in cases:
        print(f"\n=== Running case: {case['name']} (robot: {robot_type}) ===")
        run_case(
            robot=robot,
            traj_follow_jit=traj_follow_jit,
            world_coll_config_path=case["obstacle"],
            traj_type=case["traj"],
            letters=case["letters"],
            start_handle_position=case["start_pos"],
            start_handle_wxyz=case["start_wxyz"],
            end_handle_position=case["end_pos"],
            end_handle_wxyz=case["end_wxyz"],
            timesteps=timesteps,
        )
    plot_constrain_motion_planning()
    plot_tendon_length()
    plot_error()
