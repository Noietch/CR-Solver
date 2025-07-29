import numpy as np
import matplotlib.pyplot as plt
from soul.visualization.visualizer_plot import visualize_mp_scene


def visualize_motion_planning(
    save_path: str,
    world_config_path: str,
    ax: plt.Axes,
    sample_indices: list[int] = None,
    num_pose: int = 5,
):

    try:
        data = np.load(save_path)
    except FileNotFoundError:
        print(f"Warning: Data file not found at {save_path}. Skipping plot.")
        ax.text(
            0.5,
            0.5,
            0.5,
            "Data file not found",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        return

    # Reconstruct robot states from saved data
    planned_traj_poses = data["fk_result"]
    target_position = data["target_position"]
    target_wxyz = data["target_wxyz"]
    planned_tip_traj_mat = data["planned_tip_traj"]
    planned_tip_traj = planned_tip_traj_mat[:, :3, 3]

    start_end_poses = [planned_traj_poses[0], planned_traj_poses[-1]]

    if sample_indices is None:
        num_timesteps = planned_traj_poses.shape[0]
        sample_indices = np.linspace(0, num_timesteps - 1, num_pose, dtype=int)

    selected_poses = planned_traj_poses[sample_indices]

    ax.plot(
        planned_tip_traj[:, 0], planned_tip_traj[:, 1], planned_tip_traj[:, 2], "b-"
    )

    visualize_mp_scene(
        pose=selected_poses,
        start_end_poses=start_end_poses,
        target_position=np.atleast_2d(target_position),
        target_wxyz=np.atleast_2d(target_wxyz),
        num_points=10,
        ax=ax,
        world_coll_config=world_config_path,
        save_path=save_path.replace(".npz", "_fk.png"),
    )


def plot_mp_with_coll_scene():
    fig = plt.figure(facecolor="white", figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection="3d")

    save_path = "results/test/trajopt_with_coll_sections_3_eval_0.npz"
    world_config_path = "configs/maps/mp_scene/obstacles_test.json"
    visualize_motion_planning(save_path, world_config_path, ax1, num_pose=5)

    ax1.text2D(
        0.05,
        0.95,
        "Red: Ref Traj, Blue: Planned Traj",
        transform=ax1.transAxes,
    )

    ax1.legend()
    plt.tight_layout()
    plt.savefig("results/mp_scene_examples.png")
    plt.close()


if __name__ == "__main__":
    plot_mp_with_coll_scene()
