import os

from jaxtyping import Array
import numpy as np
import matplotlib.pyplot as plt


def _plot_sphere(ax, center, radius):

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
        color=np.random.rand(3),
        alpha=0.6,
    )


def visualize_pcc_model_2d(
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
        colors = ["r", "g", "b", "y", "m", "c"]
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


def visualize_pcc_model_3d(
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
    positions = transform[:, :3, 3]

    ax = plt.figure().add_subplot(projection="3d")
    batch_size = transform.shape[0]
    for i in range(batch_size):
        positions = transform[i, :, :3, 3]
        colors = ["r", "g", "b", "y", "m", "c"]
        if num_points is not None:
            for i in range(len(positions) // num_points):
                ax.plot(
                    positions[i * num_points : (i + 1) * num_points, 0],
                    positions[i * num_points : (i + 1) * num_points, 1],
                    positions[i * num_points : (i + 1) * num_points, 2],
                    c=colors[i % len(colors)],
                    linewidth=2,
                )
        else:
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                c="black",
                linewidth=3,
            )

    if target_position is not None:
        # draw target position
        ax.scatter(
            target_position[:, 0],
            target_position[:, 1],
            target_position[:, 2],
            c="red",
            marker="x",
        )

        # draw obstacle spheres
        # if goal.obstacle_sphere is not None:
        #     for sphere in goal.obstacle_sphere:
        #         x, y, z, r = sphere
        #         _plot_sphere(ax, (x, y, z), r)

    # Set equal aspect ratio for 3D plot
    # Get the current axis limits
    x_lim = ax.get_xlim3d()
    y_lim = ax.get_ylim3d()
    z_lim = ax.get_zlim3d()

    # Calculate the ranges
    x_range = abs(x_lim[1] - x_lim[0])
    y_range = abs(y_lim[1] - y_lim[0])
    z_range = abs(z_lim[1] - z_lim[0])

    # Find the largest range to ensure equal scaling
    max_range = max(x_range, y_range, z_range)

    # Calculate the mid points
    x_mid = (x_lim[1] + x_lim[0]) / 2
    y_mid = (y_lim[1] + y_lim[0]) / 2
    z_mid = (z_lim[1] + z_lim[0]) / 2

    # Add padding
    padding = 0.1 * max_range

    # Set new limits with equal scaling
    ax.set_xlim3d(x_mid - max_range / 2 - padding, x_mid + max_range / 2 + padding)
    ax.set_ylim3d(y_mid - max_range / 2 - padding, y_mid + max_range / 2 + padding)
    ax.set_zlim3d(z_mid - max_range / 2 - padding, z_mid + max_range / 2 + padding)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    if save_path is not None:
        plt.savefig(save_path)
