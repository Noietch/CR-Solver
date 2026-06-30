import os

import jax

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

import time
from typing import Sequence

import jax.numpy as jnp
import jaxlie
import numpy as np
import viser

from soul.geom import CollGeom, RobotCollision, WorldCollision
from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.solver.traj_optimizer import TrajOptimizer
from soul.visualization.visualizer_viser import ViserSoftRobot, ViserWorld

DISABLE_JIT = False

if DISABLE_JIT:
    import os

    import jax

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


def viser_main():
    world_coll_config_path = (
        "configs/maps/constrain_motion_planning/obstacles_con_A.json"
    )
    # Setup Environment
    robot = CCRobot.from_config("configs/robots/cc_con_eval.json")
    robot_coll = RobotCollision.from_config("configs/robots/cc_con_eval.json")
    world_coll = WorldCollision.from_config(world_coll_config_path)

    vmapped_is_trajectory_in_collision = jax.vmap(
        is_trajectory_in_collision, in_axes=(0, None, None, None)
    )

    # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(
        server, robot, robot_coll, root_node_name="/robot"
    )
    robot_vis.create_robot_visualizations()
    obstacles_vis = ViserWorld(
        server,
        world_coll,
        is_handle_able=True,
        config_path=world_coll_config_path
    )
    obstacles_vis.create_mesh_visualizations()
    """
    for square
    start_handle_position = (-0.05, -2.3, 0.96)
    start_handle_wxyz = (0.83, 0.54, 0.06, 0.1)
    end_handle_position = (0.7, -1.45, 2.75)
    end_handle_wxyz = (1.0, 0, 0, 0)

    for sine
    start_handle_position = (-0.01, -2.51, 0.76)
    start_handle_wxyz = (0.82, 0.47, 0.25, 0.2)
    end_handle_position = (1.23, -1.25, 2.75)
    end_handle_wxyz = (1.0, 0, 0, 0)

    Handle settings for words
    for letter "A"
    start_handle_position = (-0.01, -2.51, 0.76)
    start_handle_wxyz = (0.82, 0.47, 0.25, 0.2)
    end_handle_position = (1.23, -1.25, 2.75)
    end_handle_wxyz = (1.0, 0, 0, 0)

    for letter "R"
    start_handle_position = (-0.11, -2.36, 0.87)
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
        "/start",
        scale=0.3,
        position=start_handle_position,
        wxyz=start_handle_wxyz
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
        linear_button = server.gui.add_button(
            "Plan Linear Traj", disabled=False
        )
        square_button = server.gui.add_button(
            "Plan Square Traj", disabled=False
        )
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
    traj_solver = TrajOptimizer(robot, robot_coll, timesteps)
    traj_follow_jit = jax.jit(traj_solver.optimize_tip_traj_follow)

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
            reference_traj,
            color=np.array([1.0, 0.0, 0.0]),
            name="reference_traj"
        )

        print("Start planning....")
        # Update obstacle information from the visualizer
        start_time = time.time()
        cfg = traj_follow_jit(
            reference_traj, [obstacles_vis.world_coll.obstacles]
        )
        is_in_collision = vmapped_is_trajectory_in_collision(
            cfg, robot, robot_coll, [obstacles_vis.world_coll.obstacles]
        )
        jax.block_until_ready(is_in_collision)
        end_time = time.time()
        if is_in_collision.any():
            print("Trajectory is in collision!")
        planning_time = end_time - start_time
        print(f"Planning finished in {planning_time:.5f} seconds.")

        global_traj = robot.forward_kinematics(cfg)

        robot_vis.visualize_traj_collisions(robot, cfg)
        robot_vis.visualize_tip_traj(
            global_traj, color=np.array([0.0, 0.0, 1.0]), name="planned_traj"
        )

        for i in range(timesteps):
            time.sleep(0.01)
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
