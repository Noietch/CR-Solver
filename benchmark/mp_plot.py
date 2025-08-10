import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib  # Import matplotlib for rcParams
from soul.visualization.visualizer_plot import visualize_cc_model_3d, visualize_mp_scene

TICK_LABELSIZE = 18
TICK_PAD = 8


def set_3d_tick_labelsize(
    ax: plt.Axes, size: int = TICK_LABELSIZE, pad: int = TICK_PAD
) -> None:
    ax.xaxis.set_tick_params(labelsize=size, pad=pad)
    ax.yaxis.set_tick_params(labelsize=size, pad=pad)
    if hasattr(ax, "zaxis"):
        ax.zaxis.set_tick_params(labelsize=size, pad=pad)


def visualize_motion_planning(
    save_dir: str,
    file_name: str,
    world_config_path: str,
    ax: plt.Axes,
    sample_indices: list[int] = None,
    num_pose: int = 5,
):
    file_path = f"{save_dir}/{file_name}"
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Warning: Data file not found at {file_path}. Skipping plot.")
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

    save_path = os.path.join(
        save_dir, "result_plot", file_name.replace(".npz", "_fk.png")
    )
    visualize_mp_scene(
        pose=selected_poses,
        start_end_poses=start_end_poses,
        target_position=np.atleast_2d(target_position),
        target_wxyz=np.atleast_2d(target_wxyz),
        num_points=10,
        ax=ax,
        world_coll_config=world_config_path,
        save_path=save_path,
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

    visualize_cc_model_3d(
        pose=selected_poses,
        num_points=10,
        ax=ax,
        world_coll_config=world_config_path,
        save_path=save_path.replace(".npz", "_fk.png"),
    )


    ax.scatter(
        ref_traj_pos[:, 0],
        ref_traj_pos[:, 1],
        ref_traj_pos[:, 2],
        c="#7ea6e0",
        marker="o",
        s=8,
        label="Reference",
    )

    ax.scatter(
        planned_tip_pos[:, 0],
        planned_tip_pos[:, 1],
        planned_tip_pos[:, 2],
        c="#ea6b66",
        marker="o",
        s=6,
        label="Experiment",
    )

    # ax.plot(ref_traj_pos[:, 0], ref_traj_pos[:, 1], ref_traj_pos[:, 2], c="#7ea6e0", linewidth=4, label="Reference")
    # ax.plot(planned_tip_pos[:, 0], planned_tip_pos[:, 1], planned_tip_pos[:, 2], c="#ea6b66", linewidth=8, label="Experiment")

    ax.patch.set_alpha(0.0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")
    # ax.legend(frameon=False, markerscale=4)

    # ax.plot(ref_traj_pos[:, 0], ref_traj_pos[:, 1], ref_traj_pos[:, 2], "r--", linewidth=5 )
    # ax.plot(planned_tip_pos[:, 0], planned_tip_pos[:, 1], planned_tip_pos[:, 2], "b-", linewidth=5)
    save_path = save_path.replace(".npz", "_plot.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=600)

    print(
        f"--- Trajectory Metrics for Letter {world_config_path.split('configs/maps/obstacles_con_')[-1].split('.')[0]} ---"
    )
    print(f"Position Error (mean): {data['pos_error_mean'] * 1000:.4f}mm")
    print(f"Position Error (std):  {data['pos_error_std'] * 1000:.4f}mm")
    print(f"Rotation Error (mean): {np.rad2deg(data['rot_error_mean']):.4f}deg")
    print(f"Rotation Error (std):  {np.rad2deg(data['rot_error_std']):.4f}deg")
    print(f"Planning Time: {data['planning_time'] * 1000:.4f}ms")


def visualize_constrain_motion_planning_traj(
    save_path: str,
    ax: plt.Axes,
    sample_indices: list[int] = None,
):
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

    single_save_path = save_path.replace(".npz", "_fk_traj.png")
    ax.set_xlim3d(-1.2, 0.2)
    ax.set_ylim3d(-0.85, 0.5)
    ax.set_zlim3d(-0.9, 0.7)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Reduce tick density
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(3))

    ax.grid(False)

    if single_save_path is not None:
        plt.savefig(single_save_path, bbox_inches="tight", dpi=300)

    # ax.scatter(ref_traj_pos[:, 0], ref_traj_pos[:, 1], ref_traj_pos[:, 2], c="#7ea6e0", marker="o",s=8, label="Reference")
    ax.scatter(
        planned_tip_pos[:, 0],
        planned_tip_pos[:, 1],
        planned_tip_pos[:, 2],
        c="#ea6b66",
        marker="o",
        s=27,
        label="Experiment",
    )

    # ax.plot(ref_traj_pos[:, 0], ref_traj_pos[:, 1], ref_traj_pos[:, 2], c="#7ea6e0", linewidth=4, label="Reference")
    # ax.plot(planned_tip_pos[:, 0], planned_tip_pos[:, 1], planned_tip_pos[:, 2], c="#ea6b66", linewidth=8, label="Experiment")

    ax.patch.set_alpha(0.0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")
    # ax.legend(frameon=False, markerscale=4)

    # ax.plot(ref_traj_pos[:, 0], ref_traj_pos[:, 1], ref_traj_pos[:, 2], "r--", linewidth=5 )
    # ax.plot(planned_tip_pos[:, 0], planned_tip_pos[:, 1], planned_tip_pos[:, 2], "b-", linewidth=5)
    save_path = save_path.replace(".npz", "_plot_traj.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    print(
        f"Plotted Letter {save_path.split('configs/maps/obstacles_con_')[-1].split('.')[0]}"
    )


def plot_mp_with_coll_scene(
    save_dir: str,
    file_name: str,
    world_config_path: str,
):
    fig = plt.figure(facecolor="white", figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection="3d")
    set_3d_tick_labelsize(ax1)

    visualize_motion_planning(save_dir, file_name, world_config_path, ax1, num_pose=80)

    ax1.text2D(
        0.05,
        0.95,
        "Red: Ref Traj, Blue: Planned Traj",
        transform=ax1.transAxes,
    )

    ax1.legend()
    plt.tight_layout()
    plt.savefig("results/mp_scene_examples.png", bbox_inches="tight", dpi=800)
    plt.close()


def plot_constrain_motion_planning():
    fig = plt.figure(facecolor="white", figsize=(24, 12))
    ax1 = fig.add_subplot(241, projection="3d")
    ax2 = fig.add_subplot(242, projection="3d")
    ax3 = fig.add_subplot(243, projection="3d")
    ax4 = fig.add_subplot(244, projection="3d")
    ax5 = fig.add_subplot(245, projection="3d")
    ax6 = fig.add_subplot(246, projection="3d")
    ax7 = fig.add_subplot(247, projection="3d")
    ax8 = fig.add_subplot(248, projection="3d")

    for _ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8):
        set_3d_tick_labelsize(_ax)

    visualize_constrain_motion_planning(
        save_path="results/traj_following/get_icra_traj_I/get_icra_traj.npz",
        world_config_path="configs/maps/constrain_motion_planning/obstacles_con_I.json",
        ax=ax1,
        num_pose=15,
    )
    visualize_constrain_motion_planning(
        save_path="results/traj_following/get_icra_traj_C/get_icra_traj.npz",
        world_config_path="configs/maps/constrain_motion_planning/obstacles_con_C.json",
        ax=ax2,
        num_pose=15,
    )
    visualize_constrain_motion_planning(
        save_path="results/traj_following/get_icra_traj_R/get_icra_traj.npz",
        world_config_path="configs/maps/constrain_motion_planning/obstacles_con_R.json",
        ax=ax3,
        num_pose=15,
    )
    visualize_constrain_motion_planning(
        save_path="results/traj_following/get_icra_traj_A/get_icra_traj.npz",
        world_config_path="configs/maps/constrain_motion_planning/obstacles_con_A.json",
        ax=ax4,
        num_pose=15,
    )

    visualize_constrain_motion_planning_traj(
        save_path="results/traj_following/get_icra_traj_I/get_icra_traj.npz",
        ax=ax5,
    )
    visualize_constrain_motion_planning_traj(
        save_path="results/traj_following/get_icra_traj_C/get_icra_traj.npz",
        ax=ax6,
    )
    visualize_constrain_motion_planning_traj(
        save_path="results/traj_following/get_icra_traj_R/get_icra_traj.npz",
        ax=ax7,
    )
    visualize_constrain_motion_planning_traj(
        save_path="results/traj_following/get_icra_traj_A/get_icra_traj.npz",
        ax=ax8,
    )

    plt.tight_layout()
    save_path = "results/motion_planning_examples.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    print(f"Saved plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    # === FONT CONFIGURATION (APPLY GLOBALLY) ===
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    # ==========================================

    # save_dir = "results/13.pick_from_shelf"
    # world_config_path = "configs/maps/mp_scene/obstacles_13.pick_from_shelf.json"

    # error_path_list = []
    # for save_path in os.listdir(save_dir):
    #     if save_path.endswith(".npz") and "all_trials_results" not in save_path:
    #         try:
    #             plot_mp_with_coll_scene(
    #                 save_dir=save_dir,
    #                 file_name=save_path,
    #                 world_config_path=world_config_path,
    #             )
    #         except Exception as e:
    #             error_path_list.append(save_path)
    #             print(f"Error in plot_mp_with_coll_scene: {e}, {save_path}")
    # print("--------------ERROR PATH LIST--------------")
    # print(error_path_list)
    plot_constrain_motion_planning()
