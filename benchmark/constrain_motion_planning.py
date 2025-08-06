import jax
import jax.numpy as jnp
import jaxlie
import time
import viser
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from soul.robots.cc_robot import CCRobot
from soul.robots.cc_robot_extend import CCRobot as CCRobotExtend
from soul.geom import RobotCollision, WorldCollision
from soul.solver import ConstrainedMotionPlanner
from soul.visualization.visualizer_viser import ViserSoftRobot, ViserWorld
from benchmark.mp_plot import visualize_constrain_motion_planning

# TODO: 移动到example，只保留运行代码
# TODO: 美化图像

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


LETTER_HEIGHT = 2.5
LETTER_WIDTH = 1.5


def create_I():
    """Generates path points for the letter 'I'."""
    return [(LETTER_WIDTH / 2, 0), (LETTER_WIDTH / 2, LETTER_HEIGHT)]


def create_C():
    """Generates path points for a curved letter 'C'."""
    return [
        (
            LETTER_WIDTH / 2 + (LETTER_WIDTH / 2) * np.cos(t),
            LETTER_HEIGHT / 2 + (LETTER_HEIGHT / 2) * np.sin(t),
        )
        for t in np.linspace(0.4 * np.pi, 1.6 * np.pi, 20)
    ]


def create_R():
    """Generates path points for curved letter 'R'."""
    stem = [(0, 0), (0, LETTER_HEIGHT)]
    curve_center_y = LETTER_HEIGHT * 0.75
    curve_radius_y = LETTER_HEIGHT * 0.25
    curve_radius_x = LETTER_WIDTH
    curve = [
        (curve_radius_x * np.cos(t), curve_center_y + curve_radius_y * np.sin(t))
        for t in np.linspace(np.pi / 2, -np.pi / 3, 15)
    ]
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
        (LETTER_WIDTH * 0.3, LETTER_HEIGHT / 3),
        (LETTER_WIDTH * 0.8, LETTER_HEIGHT / 3),
    ]


def get_icra_traj(
    start_position: np.ndarray,
    start_wxyz: np.ndarray,
    end_position: np.ndarray,
    end_wxyz: np.ndarray,  # Unused but kept for consistent signature
    timesteps: int,
) -> jaxlie.SE3:
    """
    Generates a 3D "ICRA" trajectory starting at the start_handle's pose.
    The width is controlled by the x-difference between handles.
    The height is controlled by the y-difference between handles.
    """
    # word_funcs = {'I': create_I, 'C': create_C, 'R': create_R, 'A': create_A}
    word_funcs = {"A": create_A}
    word_str = "ICRA"
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
    local_points_3d = np.hstack(
        [scaled_path_2d, np.zeros((scaled_path_2d.shape[0], 1))]
    )

    # 5. Define the world transformation from the start handle's pose
    start_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=start_wxyz), translation=start_position
    )

    # 6. Apply the transformation to map local points to world space
    world_points = start_pose.apply(local_points_3d)

    # 7. Resample the final 3D path for a smooth trajectory
    distances = np.cumsum(np.linalg.norm(np.diff(world_points, axis=0), axis=1))
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
        A dictionary with position and rotation error stats (mean and std).
    """
    # Position error
    ref_pos = reference_traj.translation()
    plan_pos = planned_traj.translation()
    position_errors = jnp.linalg.norm(plan_pos - ref_pos, axis=-1)

    # Rotation error
    ref_rot = reference_traj.rotation()
    plan_rot = planned_traj.rotation()
    rotation_errors = jnp.linalg.norm((ref_rot.inverse() @ plan_rot).log(), axis=-1)

    metrics = {
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
    points_2d = np.array(create_square())
    handle_diff = np.array(end_position) - np.array(start_position)
    target_side_length = max(abs(handle_diff[0]), 0.1)

    natural_side_length = points_2d[:, 0].max() - points_2d[:, 0].min()
    scale = (
        target_side_length / natural_side_length if natural_side_length > 1e-6 else 1.0
    )
    scaled_path_2d = points_2d * scale

    local_points_3d = np.hstack(
        [scaled_path_2d, np.zeros((scaled_path_2d.shape[0], 1))]
    )
    start_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=start_wxyz), translation=start_position
    )
    world_points = start_pose.apply(local_points_3d)

    distances = np.cumsum(np.linalg.norm(np.diff(world_points, axis=0), axis=1))
    distances = np.insert(distances, 0, 0)
    new_distances = np.linspace(0, distances[-1], timesteps)

    interp_coords = [
        np.interp(new_distances, distances, world_points[:, i]) for i in range(3)
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


def viser_main():
    world_coll_config_path = (
        "configs/maps/constrain_motion_planning/obstacles_con_A.json"
    )
    # Setup Environment
    robot = CCRobot.from_config("configs/robots/cc_con_eval.json")
    robot_coll = RobotCollision.from_config("configs/robots/cc_con_eval.json")
    world_coll = WorldCollision.from_config(world_coll_config_path)

    # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot, robot_coll, root_node_name="/robot")
    robot_vis.create_robot_visualizations()
    obstacles_vis = ViserWorld(
        server, world_coll, is_handle_able=True, config_path=world_coll_config_path
    )
    obstacles_vis.create_mesh_visualizations()
    """  
    for square
    start_handle_position = (-0.01, -2.46, 0.9)
    start_handle_wxyz = (0.83, 0.54, 0.06, 0.1)
    end_handle_position = (0.76, -1.31, 2.75)
    end_handle_wxyz = (1.0, 0, 0, 0)

    for sine
    start_handle_position = (0.5, -2.17, 0.89)
    start_handle_wxyz = (1, 0, 0, 0)
    end_handle_position = (0.51, -2.17, 1.49)
    end_handle_wxyz = (1.0, 0, 0, 0)
    
    Handle settings for words
    for letter "A"
    start_handle_position = (-0.01, -2.51, 0.76)
    start_handle_wxyz = (0.82, 0.47, 0.25, 0.2)
    end_handle_position = (1.23, -1.25, 2.75)
    end_handle_wxyz = (1.0, 0, 0, 0)
    
    for letter "R"
    start_handle_position = (-0.05, -2.45, 0.93)
    start_handle_wxyz = (0.85, 0.49, 0.15, 0.14)
    end_handle_position = (0.79, -1.26, 2.75)
    end_handle_wxyz = (1.0, 0, 0, 0)
    
    for letter "C"
    start_handle_position = (-0.01, -2.46, 0.9)
    start_handle_wxyz = (0.83, 0.54, 0.06, 0.1)
    end_handle_position = (0.75, -1.31, 2.75)
    end_handle_wxyz = (1.0, 0, 0, 0)
    
    for letter "I"
    start_handle_position = (-0.01, -2.45, 0.91)
    start_handle_wxyz = (0.83, 0.54, 0.06, 0.1)
    end_handle_position = (0.75, -1.37, 2.75)
    end_handle_wxyz = (1.0, 0, 0, 0)
    """

    start_handle_position = (-0.01, -2.51, 0.76)
    start_handle_wxyz = (0.82, 0.47, 0.25, 0.2)
    end_handle_position = (1.23, -1.25, 2.75)
    end_handle_wxyz = (1.0, 0, 0, 0)
    radius = robot.config.length * robot.config.num_sections

    # Setup GUI
    start_handle = server.scene.add_transform_controls(
        "/start", scale=0.3, position=start_handle_position, wxyz=start_handle_wxyz
    )
    end_handle = server.scene.add_transform_controls(
        "/end", scale=0.3, position=end_handle_position, wxyz=end_handle_wxyz
    )
    server.scene.add_icosphere(
        "/target_sphere",
        radius=radius,
        color=(1.0, 0.8, 0.8),
        position=(0.0, 0.0, 0.0),
    )

    with server.gui.add_folder("Trajectory Controls"):
        linear_button = server.gui.add_button("Plan Linear Traj", disabled=False)
        square_button = server.gui.add_button("Plan Square Traj", disabled=False)
        sine_button = server.gui.add_button("Plan Sine Traj", disabled=False)
        icra_button = server.gui.add_button("Plan ICRA Traj", disabled=False)
        replay_button = server.gui.add_button("Replay", disabled=True)

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

    # Set up trajopt parameters
    timesteps = 150
    traj_solver = ConstrainedMotionPlanner(robot, robot_coll, timesteps)
    traj_follow_jit = jax.jit(traj_solver.tip_traj_follow)

    global_traj = None

    def plan_and_visualize(ref_traj_func):
        """Generic function to plan and visualize a trajectory."""
        print(f"Generating trajectory with {ref_traj_func.__name__}...")
        nonlocal global_traj
        replay_button.disabled = True

        reference_traj = ref_traj_func(
            start_handle.position,
            start_handle.wxyz,
            end_handle.position,
            end_handle.wxyz,
            timesteps,
        )
        robot_vis.visualize_tip_traj(
            reference_traj, color=np.array([1.0, 0.0, 0.0]), name="reference_traj"
        )

        print("Start planning....")
        # Update obstacle information from the visualizer
        start_time = time.time()
        cfg = traj_follow_jit(reference_traj, [obstacles_vis.world_coll.obstacles])
        jax.block_until_ready(cfg)
        end_time = time.time()
        planning_time = end_time - start_time
        print(f"Planning finished in {planning_time:.4f} seconds.")

        global_traj = robot.forward_kinematics(cfg)

        planned_tip_traj = jaxlie.SE3.from_matrix(global_traj[:, -1, :, :])

        calculate_traj_metrics_jit = jax.jit(calculate_traj_metrics)
        # Calculate and print metrics
        metrics = calculate_traj_metrics_jit(reference_traj, planned_tip_traj)
        print("--- Trajectory Metrics ---")
        print(f"Position Error (mean): {metrics['pos_error_mean']:.4f}m")
        print(f"Position Error (std):  {metrics['pos_error_std']:.4f}m")
        print(f"Rotation Error (mean): {metrics['rot_error_mean']:.4f}rad")
        print(f"Rotation Error (std):  {metrics['rot_error_std']:.4f}rad")
        print(f"Planning Time: {planning_time:.4f}s")

        # Save results
        save_dir = os.path.join("results", "traj_following", ref_traj_func.__name__)
        os.makedirs(save_dir, exist_ok=True)

        # Save reference trajectory to CSV
        ref_csv_filename = f"{ref_traj_func.__name__}_reference.csv"
        ref_csv_save_path = os.path.join(save_dir, ref_csv_filename)
        save_ref_traj_to_csv(reference_traj, ref_csv_save_path)

        filename = f"{ref_traj_func.__name__}.npz"
        save_path = os.path.join(save_dir, filename)

        # Get obstacle information for saving.
        obstacles_for_saving = obstacles_vis.world_coll.obstacles

        if isinstance(robot, CCRobotExtend):
            np.savez(
                save_path,
                solution_states_theta=np.asarray(cfg.theta),
                solution_states_phi=np.asarray(cfg.phi),
                solution_states_length=np.asarray(cfg.length),
                obstacles=obstacles_for_saving,
                fk_result=np.asarray(global_traj),
                target_position=np.asarray(reference_traj.translation()),
                target_wxyz=np.asarray(reference_traj.rotation().wxyz),
                planned_tip_traj=np.asarray(planned_tip_traj.as_matrix()),
                planning_time=planning_time,
                **{k: np.array(v) for k, v in metrics.items()},
            )
        else:
            np.savez(
                save_path,
                solution_states_theta=np.asarray(cfg.theta),
                solution_states_phi=np.asarray(cfg.phi),
                obstacles=obstacles_for_saving,
                fk_result=np.asarray(global_traj),
                target_position=np.asarray(reference_traj.translation()),
                target_wxyz=np.asarray(reference_traj.rotation().wxyz),
                planned_tip_traj=np.asarray(planned_tip_traj.as_matrix()),
                planning_time=planning_time,
                **{k: np.array(v) for k, v in metrics.items()},
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
        save_path = "results/motion_planning_examples.png"
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()

        robot_vis.visualize_traj_collisions(robot, cfg)
        robot_vis.visualize_tip_traj(
            global_traj, color=np.array([0.0, 0.0, 1.0]), name="planned_traj"
        )

        for i in range(timesteps):
            time.sleep(0.02)
            robot_vis.update_pose(global_traj[i])

        replay_button.disabled = False
        print("Animation finished. Ready to replay or plan new trajectory.")

    def on_handle_update(handle: viser.TransformControlsHandle):
        """Update GUI when handles are moved."""
        start_pos_text.value = str(np.round(start_handle.position, 2))
        start_wxyz_text.value = str(np.round(start_handle.wxyz, 2))
        end_pos_text.value = str(np.round(end_handle.position, 2))
        end_wxyz_text.value = str(np.round(end_handle.wxyz, 2))

    def replay_callback(_: viser.GuiButtonHandle):
        nonlocal global_traj
        if global_traj is None:
            print("No trajectory to replay.")
            return
        print("Replaying last trajectory...")
        for i in range(timesteps):
            time.sleep(0.02)
            robot_vis.update_pose(global_traj[i])
        print("Replay finished.")

    linear_button.on_click(lambda _: plan_and_visualize(get_linear_traj))
    square_button.on_click(lambda _: plan_and_visualize(get_square_traj))
    sine_button.on_click(lambda _: plan_and_visualize(get_sine_traj))
    icra_button.on_click(lambda _: plan_and_visualize(get_icra_traj))
    replay_button.on_click(replay_callback)

    start_handle.on_update(on_handle_update)
    end_handle.on_update(on_handle_update)

    on_handle_update(start_handle)

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    viser_main()
