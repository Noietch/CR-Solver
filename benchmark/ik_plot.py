import numpy as np
import matplotlib.pyplot as plt

from soul.visualization.visualizer_plot import visualize_cc_model_3d


def visualize_ik_with_coll(
    save_path: str, world_config_path: str, ax: plt.Axes, selected_indices: list[int]
):
    data = np.load(save_path)
    target_position = data["target_position"]
    target_wxyz = data["target_wxyz"]
    fk_result = data["fk_result"]
    # Randomly select 3 solutions
    num_solutions = len(fk_result)
    if selected_indices is None:
        selected_indices = np.random.choice(num_solutions, size=3, replace=False)
    # Get the selected solutions
    print(f"selected_indices: {selected_indices}")
    fk_result = fk_result[selected_indices]
    target_position = target_position[selected_indices]
    target_wxyz = target_wxyz[selected_indices]
    visualize_cc_model_3d(
        pose=fk_result,
        num_points=10,
        target_wxyz=target_wxyz,
        target_position=target_position,
        world_coll_config=world_config_path,
        save_path=save_path.replace(".npz", "_fk.png"),
        ax=ax,
    )


def visualize_constrain_motion_planning(
    save_path: str,
    world_config_path: str,
    ax: plt.Axes,
    sample_indices: list[int] = None,
    num_pose: int = 5,
):
    """
    Visualizes a planned trajectory from a saved data file.

    Args:
        save_path: Path to the .npz file with trajectory data.
        world_config_path: Path to the world collision configuration file.
        ax: The matplotlib 3D axes to plot on.
        title: The title for the subplot.
        sample_indices: A list of frame indices to visualize the robot's body.
                        If None, a default sampling will be used.
    """
    try:
        data = np.load(save_path, allow_pickle=True)
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
    num_timesteps = data["solution_states_theta"].shape[0]
    planned_traj_poses = data["fk_result"]
    ref_traj_pos = data["target_position"]
    planned_tip_traj_mat = data["planned_tip_traj"]
    planned_tip_pos = planned_tip_traj_mat[:, :3, 3]

    if sample_indices is None:
        num_timesteps = planned_traj_poses.shape[0]
        sample_indices = np.linspace(0, num_timesteps - 1, num_pose, dtype=int)

    selected_poses = planned_traj_poses[sample_indices]

    ax.plot(ref_traj_pos[:, 0], ref_traj_pos[:, 1], ref_traj_pos[:, 2], "r--")
    ax.plot(planned_tip_pos[:, 0], planned_tip_pos[:, 1], planned_tip_pos[:, 2], "b-")

    visualize_cc_model_3d(
        pose=selected_poses,
        num_points=10,
        ax=ax,
        world_coll_config=world_config_path,
        save_path=save_path.replace(".npz", "_fk.png"),
    )

    print(
        f"--- Trajectory Metrics for Letter {world_config_path.split('configs/maps/obstacles_con_')[-1].split('.')[0]} ---"
    )
    print(f"Position Error (mean): {data['pos_error_mean'] * 1000:.4f}mm")
    print(f"Position Error (std):  {data['pos_error_std'] * 1000:.4f}mm")
    print(f"Rotation Error (mean): {np.rad2deg(data['rot_error_mean']):.4f}deg")
    print(f"Rotation Error (std):  {np.rad2deg(data['rot_error_std']):.4f}deg")


def plot_ik_with_coll():
    fig = plt.figure(facecolor="white", figsize=(8, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    visualize_ik_with_coll(
        "results/ik_with_coll_lattice/ik_with_coll_sections_3_eval_100.npz",
        "configs/maps/ik_maps/obstacles_lattice.json",
        ax=ax1,
        selected_indices=[15, 16],
    )
    visualize_ik_with_coll(
        "results/ik_with_coll_icosahedron/ik_with_coll_sections_6_eval_100.npz",
        "configs/maps/ik_maps/obstacles_icosahedron.json",
        ax=ax2,
        selected_indices=[83, 32, 90],
    )
    plt.tight_layout()
    plt.savefig("results/ik_examples.png")
    plt.close()


def plot_constrain_motion_planning():
    fig = plt.figure(facecolor="white", figsize=(24, 6))
    ax1 = fig.add_subplot(141, projection="3d")
    ax2 = fig.add_subplot(142, projection="3d")
    ax3 = fig.add_subplot(143, projection="3d")
    ax4 = fig.add_subplot(144, projection="3d")

    visualize_constrain_motion_planning(
        save_path="results/traj_following/get_icra_traj_I/get_icra_traj.npz",
        world_config_path="configs/maps/obstacles_con_I.json",
        ax=ax1,
    )
    visualize_constrain_motion_planning(
        save_path="results/traj_following/get_icra_traj_C/get_icra_traj.npz",
        world_config_path="configs/maps/obstacles_con_C.json",
        ax=ax2,
    )
    visualize_constrain_motion_planning(
        save_path="results/traj_following/get_icra_traj_R/get_icra_traj.npz",
        world_config_path="configs/maps/obstacles_con_R.json",
        ax=ax3,
    )
    visualize_constrain_motion_planning(
        save_path="results/traj_following/get_icra_traj_A/get_icra_traj.npz",
        world_config_path="configs/maps/obstacles_con_A.json",
        ax=ax4,
        num_pose=8,
    )

    plt.tight_layout()
    save_path = "results/motion_planning_examples.png"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    plot_constrain_motion_planning()
