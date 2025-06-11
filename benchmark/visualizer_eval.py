import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import jax
from jax import Array
from jaxtyping import Float
import os


def create_figure() -> Axes:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("PCC Model Visualization")
    return ax

def draw_pcc_3d(
    ax: Axes,  
    pose: jax.Array,
    target_position: jax.Array = None,
    num_points: int = None,
):
    """
    dram the points samples
    """
    if pose.ndim == 3:
        pose = pose[None, :, :, :]
    if target_position is not None and target_position.ndim == 1:
        target_position = target_position[None, :]

    # batch_size = pose.shape[0]
    # for i in range(batch_size):
    #     positions = pose[i, :, :3, 3]
    #     segment_colors = ["r", "g", "b", "y", "m", "c"]
    #     if num_points is not None:
    #         for i in range(len(positions) // num_points):
    #             ax.plot(
    #                 positions[i * num_points : (i + 1) * num_points, 0],
    #                 positions[i * num_points : (i + 1) * num_points, 1],
    #                 positions[i * num_points : (i + 1) * num_points, 2],
    #                 c=segment_colors[i % len(segment_colors)],
    #                 linewidth=2,
    #             )
    #     else:
    #         ax.plot(
    #             positions[:, 0],
    #             positions[:, 1],
    #             positions[:, 2],
    #             c="black",
    #             linewidth=3,
    #         )

    if target_position is not None:
        ax.scatter(
            target_position[:, 0],
            target_position[:, 1],
            target_position[:, 2],
            c="red",
            marker="x",
            s=100
        )

def finalize_plot_3d(ax:Axes, save_path: str = None):
    x_lim, y_lim, z_lim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    x_range, y_range, z_range = abs(x_lim[1] - x_lim[0]), abs(y_lim[1] - y_lim[0]), abs(z_lim[1] - z_lim[0])
    max_range = max(x_range, y_range, z_range)
    x_mid, y_mid, z_mid = (x_lim[1] + x_lim[0]) / 2, (y_lim[1] + y_lim[0]) / 2, (z_lim[1] + z_lim[0]) / 2
    
    padding = 0.1 * max_range
    ax.set_xlim3d(x_mid - max_range / 2 - padding, x_mid + max_range / 2 + padding)
    ax.set_ylim3d(y_mid - max_range / 2 - padding, y_mid + max_range / 2 + padding)
    ax.set_zlim3d(z_mid - max_range / 2 - padding, z_mid + max_range / 2 + padding)
    ax.set_box_aspect([1, 1, 1])

    # save figure
    if save_path:
        dir_path = os.path.dirname(save_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path)
        print(f"figure already saved: {save_path}")
    
 
def visualizer_forward_samples(ax: Axes, batch_pose: Float[Array, "*batch num_sections 4 4"], batch_target_position:jax.Array, num_points:int, save_path:str ):
    for pose, target_position in zip(batch_pose, batch_target_position):
        draw_pcc_3d(ax, pose, target_position, num_points)
    finalize_plot_3d(ax, save_path)
