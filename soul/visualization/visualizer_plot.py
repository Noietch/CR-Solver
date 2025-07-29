import os
import json
from jaxtyping import Array
import numpy as np
import matplotlib.pyplot as plt

from soul.geom.utils import load_mesh


def _quaternion_to_rotation_matrix(wxyz):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = wxyz
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def _plot_coordinate_frame(ax, position, rotation_matrix, scale=0.4):
    """Plot coordinate frame with x, y, z axes."""
    # Define unit vectors
    origin = position
    x_axis = rotation_matrix @ np.array([scale, 0, 0])
    y_axis = rotation_matrix @ np.array([0, scale, 0])
    z_axis = rotation_matrix @ np.array([0, 0, scale])

    # Plot axes
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        x_axis[0],
        x_axis[1],
        x_axis[2],
        color="red",
        arrow_length_ratio=0.1,
        linewidth=2,
        label="X",
    )
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        y_axis[0],
        y_axis[1],
        y_axis[2],
        color="green",
        arrow_length_ratio=0.1,
        linewidth=2,
        label="Y",
    )
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        z_axis[0],
        z_axis[1],
        z_axis[2],
        color="blue",
        arrow_length_ratio=0.1,
        linewidth=2,
        label="Z",
    )


def _plot_sphere(ax: plt.Axes, center, radius):

    # Create a mesh grid for the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the sphere
    ax.plot_surface(
        x,
        y,
        z,
        color="whitesmoke",
        alpha=0.4,
    )


def visualize_cc_model_2d(
    pose: Array,
    target_position: Array = None,
    num_points: int = None,
    save_path: str = None,
):
    if not os.path.exists(save_path):
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok=True)

    if pose.ndim == 3:
        if target_position is not None:
            target_position = target_position[None, :]
        transform = pose[None, :, :, :]
    else:
        transform = pose
    batch_size = transform.shape[0]

    for i in range(batch_size):
        positions = transform[i, :, :3, 3]
        # Plot the positions
        plt.scatter([0], [0], c="black", marker="o")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        # colors = ["r", "g", "b", "y", "m", "c"]
        colors = ["black"]
        if num_points is not None:
            for i in range(len(positions) // num_points):
                plt.plot(
                    positions[i * num_points : (i + 1) * num_points, 0],
                    positions[i * num_points : (i + 1) * num_points, 2],
                    c=colors[i % len(colors)],
                    linewidth=2,
                )
        else:
            plt.plot(
                positions[:, 0],
                positions[:, 2],
                c="black",
                linewidth=3,
            )

    if target_position is not None:
        plt.scatter(target_position[:, 0], target_position[:, 2], c="red", marker="x")

        # # draw obstacle spheres
        # if goal.obstacle_sphere is not None:
        #     for sphere in goal.obstacle_sphere:
        #         x, y, z, r = sphere
        #         circle = plt.Circle((x, z), r, color="black", fill=True)
        #         plt.gca().add_patch(circle)

    # Calculate bounds from positions
    x_max = np.max(positions[:, 0])
    x_min = np.min(positions[:, 0])
    z_max = np.max(positions[:, 2])
    z_min = np.min(positions[:, 2])

    # Consider obstacle spheres in bounds calculation if they exist
    # if goal is not None and goal.obstacle_sphere is not None:
    #     for sphere in goal.obstacle_sphere:
    #         x, z, _, r = sphere[0], sphere[2], sphere[1], sphere[3]
    #         x_max = max(x_max, x + r)
    #         x_min = min(x_min, x - r)
    #         z_max = max(z_max, z + r)
    #         z_min = min(z_min, z - r)

    # Calculate the range for both axes
    x_range = x_max - x_min
    z_range = z_max - z_min

    # Use the larger range for both axes to ensure equal scaling
    max_range = max(x_range, z_range)
    x_mid = (x_max + x_min) / 2
    z_mid = (z_max + z_min) / 2

    # Set equal aspect ratio with padding
    padding = 0.1
    plt.xlim(x_mid - max_range / 2 - padding, x_mid + max_range / 2 + padding)
    plt.ylim(z_mid - max_range / 2 - padding, z_mid + max_range / 2 + padding)
    plt.gca().set_aspect("equal")

    if save_path is not None:
        plt.savefig(save_path)


def visualize_cc_model_3d(
    pose: Array = None,
    target_wxyz: Array = None,
    target_position: Array = None,
    num_points: int = None,
    save_path: str = None,
    world_coll_config: str = None,
    ax: plt.Axes = None,
):
    if not os.path.exists(save_path):
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok=True)

    if ax is None:
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(projection="3d")

    if pose is not None:
        if pose.ndim == 3:
            if target_position is not None:
                target_position = target_position[None, :]
            transform = pose[None, :, :, :]
        else:
            transform = pose
        positions = transform[:, :3, 3]

        batch_size = transform.shape[0]
        for i in range(batch_size):
            positions = transform[i, :, :3, 3]
            # colors = ["r", "g", "b", "y", "m", "c"]
            colors = ["black"]
            if num_points is not None:
                for i in range(len(positions) // num_points):
                    ax.plot(
                        positions[i * num_points : (i + 1) * num_points, 0],
                        positions[i * num_points : (i + 1) * num_points, 1],
                        positions[i * num_points : (i + 1) * num_points, 2],
                        c=colors[i % len(colors)],
                        linewidth=2,
                    )
                for i in range(len(positions) // num_points):
                    end_point = positions[(i + 1) * num_points - 1]
                    ax.scatter(
                        end_point[0],
                        end_point[1],
                        end_point[2],
                        c=colors[i % len(colors)],
                        marker="o",
                        s=2,
                    )
            else:
                ax.plot(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    c="black",
                    linewidth=3,
                )

    # draw target orientation
    if target_wxyz is not None:
        # breakpoint()
        for i in range(len(target_wxyz)):
            rotation_matrix = _quaternion_to_rotation_matrix(target_wxyz[i])
            # Calculate appropriate scale based on scene size
            if pose is not None:
                positions = transform[0, :, :3, 3]  # Get first batch positions
                scene_range = np.max(positions, axis=0) - np.min(positions, axis=0)
                scale = np.max(scene_range) * 0.1  # 10% of scene range
            else:
                scale = 0.1  # Default scale

            # Use target_position if available, otherwise use origin
            if target_position is not None:
                frame_position = target_position[i]
            else:
                frame_position = np.array([0.0, 0.0, 0.0])  # Place at origin

            _plot_coordinate_frame(ax, frame_position, rotation_matrix, scale=scale)

    # draw obstacle spheres
    if world_coll_config is not None:
        if isinstance(world_coll_config, str):
            world_coll = json.load(open(world_coll_config))
        else:
            world_coll = world_coll_config
        for obstacle in world_coll.values():
            if obstacle["type"] == "sphere":
                x, y, z, r = *obstacle["center"], obstacle["radius"]
                _plot_sphere(ax, (x, y, z), r)
            elif obstacle["type"] == "mesh":
                decompose_type = obstacle.get("decompose_type", None)
                decompose_params = obstacle.get("decompose_params", None)
                _, original_mesh = load_mesh(
                    obstacle["path"],
                    scale=obstacle.get("scale", 1.0),
                    wxyz=obstacle.get("wxyz", [1.0, 0.0, 0.0, 0.0]),
                    position=obstacle.get("position", [0.0, 0.0, 0.0]),
                    decompose_type=decompose_type,
                    decompose_params=decompose_params,
                )
                ax.plot_trisurf(
                    original_mesh.vertices[:, 0],
                    original_mesh.vertices[:, 1],
                    original_mesh.vertices[:, 2],
                    triangles=original_mesh.faces,
                    color="gray",
                    alpha=0.6,
                )
            else:
                print(f"Unsupported obstacle type: {obstacle['type']}")

    # Set new limits with equal scaling
    ax.set_xlim3d(-1.3, 1.3)
    ax.set_ylim3d(-1.3, 1.3)
    ax.set_zlim3d(-1.3, 1.3)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Reduce tick density
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(3))

    if save_path is not None:
        plt.savefig(save_path)


def visualize_mp_scene(
    pose: Array = None,
    initial_pose: Array = None,
    target_wxyz: Array = None,
    target_position: Array = None,
    num_points: int = None,
    save_path: str = None,
    world_coll_config: str = None,
    ax: plt.Axes = None,
):
    """
    Visualizes CC robot poses for collision scenes, showing initial (green) and final (blue) states.
    """
    if save_path and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if ax is None:
        fig = plt.figure(facecolor="white")
        ax = fig.add_subplot(projection="3d")

    # Draw final poses (blue)
    if pose is not None:
        if pose.ndim == 3:
            transform = pose[None, :, :, :]
        else:
            transform = pose
        batch_size = transform.shape[0]
        for i in range(batch_size):
            positions = transform[i, :, :3, 3]
            if num_points is not None:
                num_sections = len(positions) // num_points
                for j in range(num_sections):
                    ax.plot(
                        positions[j * num_points : (j + 1) * num_points, 0],
                        positions[j * num_points : (j + 1) * num_points, 1],
                        positions[j * num_points : (j + 1) * num_points, 2],
                        c="blue",
                        linewidth=2,
                        label="Final Pose (Solution)" if i == 0 and j == 0 else None,
                    )
            else:
                ax.plot(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    c="blue",
                    linewidth=3,
                    label="Final Pose (Solution)" if i == 0 else None,
                )

    # Draw initial poses (green)
    if initial_pose is not None:
        if initial_pose.ndim == 3:
            transform = initial_pose[None, :, :, :]
        else:
            transform = initial_pose
        batch_size = transform.shape[0]
        for i in range(batch_size):
            positions = transform[i, :, :3, 3]
            if num_points is not None:
                num_sections = len(positions) // num_points
                for j in range(num_sections):
                    ax.plot(
                        positions[j * num_points : (j + 1) * num_points, 0],
                        positions[j * num_points : (j + 1) * num_points, 1],
                        positions[j * num_points : (j + 1) * num_points, 2],
                        c="green",
                        linewidth=2,
                        linestyle="--",
                        label="Initial Pose (Target)" if i == 0 and j == 0 else None,
                    )
            else:
                ax.plot(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    c="green",
                    linewidth=3,
                    linestyle="--",
                    label="Initial Pose (Target)" if i == 0 else None,
                )

    # draw target orientation
    if target_wxyz is not None:
        for i in range(len(target_wxyz)):
            rotation_matrix = _quaternion_to_rotation_matrix(target_wxyz[i])
            scale = 0.1
            if pose is not None:
                temp_transform = pose
                if temp_transform.ndim == 3:
                    temp_transform = temp_transform[None, :, :, :]
                positions = temp_transform[0, :, :3, 3]
                scene_range = np.max(positions, axis=0) - np.min(positions, axis=0)
                scale = np.max(scene_range) * 0.1
            if target_position is not None:
                frame_position = target_position[i]
            else:
                frame_position = np.array([0.0, 0.0, 0.0])
            _plot_coordinate_frame(ax, frame_position, rotation_matrix, scale=scale)

    # draw obstacle spheres
    if world_coll_config is not None:
        if isinstance(world_coll_config, str):
            world_coll = json.load(open(world_coll_config))
        else:
            world_coll = world_coll_config
        for obstacle in world_coll.values():
            if obstacle["type"] == "sphere":
                x, y, z, r = *obstacle["center"], obstacle["radius"]
                _plot_sphere(ax, (x, y, z), r)
            elif obstacle["type"] == "mesh":
                decompose_type = obstacle.get("decompose_type", None)
                decompose_params = obstacle.get("decompose_params", None)
                _, original_mesh = load_mesh(
                    obstacle["path"],
                    scale=obstacle.get("scale", 1.0),
                    wxyz=obstacle.get("wxyz", [1.0, 0.0, 0.0, 0.0]),
                    position=obstacle.get("position", [0.0, 0.0, 0.0]),
                    decompose_type=decompose_type,
                    decompose_params=decompose_params,
                )
                ax.plot_trisurf(
                    original_mesh.vertices[:, 0],
                    original_mesh.vertices[:, 1],
                    original_mesh.vertices[:, 2],
                    triangles=original_mesh.faces,
                    color="gray",
                    alpha=0.6,
                )
            else:
                print(f"Unsupported obstacle type: {obstacle['type']}")

    # Set new limits with equal scaling
    ax.set_xlim3d(-1.3, 1.3)
    ax.set_ylim3d(-1.3, 1.3)
    ax.set_zlim3d(-1.3, 1.3)
    ax.set_box_aspect([1, 1, 1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(3))

    if save_path is not None:
        plt.savefig(save_path)
