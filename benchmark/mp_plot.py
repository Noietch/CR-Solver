import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib  # Import matplotlib for rcParams
from matplotlib.ticker import MultipleLocator
from soul.visualization.visualizer_plot import visualize_cc_model_3d, visualize_mp_scene

TICK_LABELSIZE = 25
TICK_PAD = 12

TICK_LABELSIZE_2D = 18
TICK_PAD_2D = 8


def set_3d_tick_labelsize(
    ax: plt.Axes, size: int = TICK_LABELSIZE, pad: int = TICK_PAD
) -> None:
    ax.xaxis.set_tick_params(labelsize=size, pad=pad)
    ax.yaxis.set_tick_params(labelsize=size, pad=pad)
    if hasattr(ax, "zaxis"):
        ax.zaxis.set_tick_params(labelsize=size, pad=pad)


def set_2d_tick_labelsize(
    ax: plt.Axes, size: int = TICK_LABELSIZE_2D, pad: int = TICK_PAD_2D
) -> None:
    ax.xaxis.set_tick_params(labelsize=size, pad=pad)
    ax.yaxis.set_tick_params(labelsize=size, pad=pad)


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
    plt.savefig(save_path, bbox_inches="tight", dpi=900)

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
        plt.savefig(single_save_path, bbox_inches="tight", dpi=900)

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
    plt.savefig(save_path, bbox_inches="tight", dpi=900)
    print(
        f"Plotted Letter {save_path.split('configs/maps/obstacles_con_')[-1].split('.')[0]}"
    )


def visualize_tendon_lengths_grid(
    csv_path: str,
    axes: list[plt.Axes],
    title: str | None = None,
) -> None:
    """
    Visualize tendon length time series grouped by section in a 1x3 grid.

    - Provided axes should be a list/array of three subplots (one per section).
    - Within each subplot, different cables (tendons) of that section are plotted with distinct colors.

    Args:
        csv_path: Path to the tendon lengths CSV. Expected header: timestep,tendon_1,...
        axes: List of three matplotlib Axes objects to draw on (columns for sections 1..3).
        title: Optional title for this 1x3 row (applied to the first subplot's title prefix).
    """
    if not os.path.exists(csv_path):
        print(f"Warning: tendon length CSV not found at {csv_path}")
        # Gracefully annotate axes
        for ax in axes[:3]:
            ax.text(
                0.5,
                0.5,
                "CSV not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
        return

    # Load CSV with header as names
    rec = np.genfromtxt(csv_path, delimiter=",", names=True)

    # Handle empty or malformed files
    if rec.size == 0:
        print(f"Warning: empty tendon length CSV at {csv_path}")
        for ax in axes[:3]:
            ax.text(
                0.5, 0.5, "Empty CSV", ha="center", va="center", transform=ax.transAxes
            )
            ax.axis("off")
        return

    names = rec.dtype.names or []
    time_key = (
        "timestep" if "timestep" in names else ("time" if "time" in names else None)
    )
    tendon_keys = [n for n in names if n != time_key]
    if len(tendon_keys) == 0:
        print(f"Warning: no tendon columns found in {csv_path}")
        for ax in axes[:3]:
            ax.text(
                0.5,
                0.5,
                "No tendon data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
        return

    # Build arrays
    if time_key is not None:
        time = np.asarray(rec[time_key])
    else:
        # Fallback to index if no time column
        time = np.arange(rec.shape[0] if hasattr(rec, "shape") else len(rec))

    lengths = np.column_stack([np.asarray(rec[k]) for k in tendon_keys])
    total_tendons = lengths.shape[1]

    # Infer number of sections from a sibling .npz file if available
    num_sections = None
    dir_name = os.path.dirname(csv_path)
    try:
        candidate_npzs = [f for f in os.listdir(dir_name) if f.endswith(".npz")]
        if len(candidate_npzs) > 0:
            npz_path = os.path.join(dir_name, candidate_npzs[0])
            with np.load(npz_path, allow_pickle=True) as data:
                if "solution_states_theta" in data:
                    theta_array = data["solution_states_theta"]
                    if theta_array.ndim >= 2:
                        num_sections = int(theta_array.shape[1])
    except Exception as e:
        print(f"Warning: failed to infer sections from npz: {e}")

    # Expect exactly 3 sections for the 1x3 grid
    if num_sections is None or num_sections <= 0:
        num_sections = 3
    # If total tendons not divisible, last section may get fewer/more; we slice safely below
    num_tendons_per_section = max(1, total_tendons // num_sections)

    color_cycle = plt.get_cmap("tab10")

    # Draw three sections (truncate/clip if fewer available)
    sections_to_plot = 3
    for sec_idx in range(sections_to_plot):
        ax = axes[sec_idx]
        start = sec_idx * num_tendons_per_section
        end = (sec_idx + 1) * num_tendons_per_section
        start = min(start, total_tendons)
        end = min(end, total_tendons)
        # Clear axis if reused
        ax.cla()
        # Plot each cable in this section
        for tendon_i, col_idx in enumerate(range(start, end)):
            color = color_cycle(tendon_i % 10)
            ax.plot(
                time,
                lengths[:, col_idx],
                color=color,
                linewidth=1.5,
                label=f"Cable {tendon_i + 1}",
            )
        # Titles and cosmetics
        ax.grid(True, linestyle=":", alpha=0.3)
        set_2d_tick_labelsize(ax)
        # if (end - start) <= 10:
        #     ax.legend(fontsize=8, frameon=False)

    # Labels (apply to leftmost and bottom row by convention; caller may refine)


def plot_mp_with_coll_scene(
    save_dir: str,
    file_name: str,
    world_config_path: str,
):
    fig = plt.figure(facecolor="white", figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection="3d")
    set_3d_tick_labelsize(ax1)

    visualize_motion_planning(save_dir, file_name, world_config_path, ax1, num_pose=80)

    plt.tight_layout()
    plt.savefig("results/mp_scene_examples.png", bbox_inches="tight", dpi=900)
    plt.close()


def plot_constrain_motion_planning():
    fig = plt.figure(facecolor="white", figsize=(16, 24))
    ax1 = fig.add_subplot(421, projection="3d")
    ax2 = fig.add_subplot(423, projection="3d")
    ax3 = fig.add_subplot(425, projection="3d")
    ax4 = fig.add_subplot(427, projection="3d")
    ax5 = fig.add_subplot(422, projection="3d")
    ax6 = fig.add_subplot(424, projection="3d")
    ax7 = fig.add_subplot(426, projection="3d")
    ax8 = fig.add_subplot(428, projection="3d")

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
    plt.savefig(save_path, bbox_inches="tight", dpi=900)
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_tendon_length() -> None:
    """
    Create a 4x3 figure for I, C, R, A letters (rows) and 3 sections (columns).
    Each row is drawn via visualize_tendon_lengths_grid over the corresponding CSV.
    """
    letters = ["I", "C", "R", "A"]
    # Create a 4x3 grid
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True)

    for row_idx, letter in enumerate(letters):
        # Expect CSV layout similar to trajectory npz sibling files
        csv_path = os.path.join(
            "results",
            "traj_following",
            f"get_icra_traj_{letter}",
            "tendon_lengths.csv",
        )
        row_axes = (
            list(axes[row_idx, :]) if isinstance(axes, np.ndarray) else axes[row_idx]
        )
        visualize_tendon_lengths_grid(
            csv_path=csv_path, axes=row_axes, title=f"Letter {letter}"
        )
        # Ensure y-axis tick spacing is 0.05 for all subplots in this row
        for ax in row_axes:
            ax.yaxis.set_major_locator(MultipleLocator(0.05))

    # Bottom row x-labels

    fig.tight_layout()
    save_path = os.path.join("results", "tendon_length_icra.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=900)
    print(f"Saved tendon length ICRA plot to {save_path}")


def visualize_error(save_path: str, ax: plt.Axes):
    """
    Plot per-timestep position and rotation errors from a saved trajectory .npz.

    - X axis: timestep index
    - Left Y axis: position error (m)
    - Right Y axis: rotation error (rad)
    """
    if not os.path.exists(save_path):
        print(f"Warning: npz not found at {save_path}")
        ax.text(
            0.5, 0.5, "NPZ not found", ha="center", va="center", transform=ax.transAxes
        )
        ax.axis("off")
        return

    try:
        data = np.load(save_path, allow_pickle=True)
    except Exception as e:
        print(f"Warning: failed to load {save_path}: {e}")
        ax.text(
            0.5, 0.5, "Load error", ha="center", va="center", transform=ax.transAxes
        )
        ax.axis("off")
        return

    # Prefer directly saved per-timestep errors if available
    if "position_errors" in data and "rotation_errors" in data:
        position_errors = np.asarray(data["position_errors"])  # (T,)
        rotation_errors = np.asarray(data["rotation_errors"])  # (T,)
        T = min(position_errors.shape[0], rotation_errors.shape[0])
        if T == 0:
            ax.text(
                0.5, 0.5, "Empty data", ha="center", va="center", transform=ax.transAxes
            )
            ax.axis("off")
            return
        position_errors = position_errors[:T]
        rotation_errors = rotation_errors[:T]
    else:
        # Fallback: compute from reference and planned poses
        if (
            "target_position" not in data
            or "target_wxyz" not in data
            or "planned_tip_traj" not in data
        ):
            print(f"Warning: required keys missing in {save_path}")
            ax.text(
                0.5,
                0.5,
                "Invalid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return
        ref_positions = np.asarray(data["target_position"])  # (T, 3)
        ref_quats_wxyz = np.asarray(data["target_wxyz"])  # (T, 4)
        tip_traj_mats = np.asarray(data["planned_tip_traj"])  # (T, 4, 4)
        T = min(ref_positions.shape[0], ref_quats_wxyz.shape[0], tip_traj_mats.shape[0])
        if T == 0:
            ax.text(
                0.5, 0.5, "Empty data", ha="center", va="center", transform=ax.transAxes
            )
            ax.axis("off")
            return
        ref_positions = ref_positions[:T]
        ref_quats_wxyz = ref_quats_wxyz[:T]
        tip_traj_mats = tip_traj_mats[:T]
        plan_positions = tip_traj_mats[:, :3, 3]
        plan_rot_mats = tip_traj_mats[:, :3, :3]
        position_errors = np.linalg.norm(plan_positions - ref_positions, axis=-1)

        def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
            w, x, y, z = q
            norm = np.sqrt(w * w + x * x + y * y + z * z)
            if norm > 0:
                w, x, y, z = w / norm, x / norm, y / norm, z / norm
            return np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )

        ref_rot_mats = np.stack(
            [quat_wxyz_to_rotmat(q) for q in ref_quats_wxyz], axis=0
        )
        R_rel = np.einsum(
            "tij,tjk->tik", np.transpose(ref_rot_mats, (0, 2, 1)), plan_rot_mats
        )
        traces = np.clip((np.trace(R_rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors = np.arccos(traces)

    # Plot
    timesteps = np.arange(len(position_errors))
    # Convert position error from meters to millimeters for plotting
    position_errors = position_errors * 1000.0

    ax.cla()
    color_pos = "#1f77b4"  # blue
    color_rot = "#ff7f0e"  # orange

    ax.plot(
        timesteps,
        position_errors,
        color=color_pos,
        linewidth=1.8,
        label="Position Error",
    )
    ax.grid(True, linestyle=":", alpha=0.3)
    set_2d_tick_labelsize(ax)

    ax_right = ax.twinx()
    ax_right.plot(
        timesteps,
        rotation_errors,
        color=color_rot,
        linewidth=1.8,
        label="Rotation Error",
    )
    set_2d_tick_labelsize(ax_right)

    title_suffix = None
    try:
        parts = os.path.normpath(save_path).split(os.sep)
        for p in parts:
            if p.startswith("get_icra_traj_") and len(p) >= len("get_icra_traj_X"):
                title_suffix = p.split("get_icra_traj_")[-1]
                break
    except Exception:
        pass


def plot_error() -> None:
    """
    Create a 4x1 vertical figure for I, C, R, A letters showing per-timestep
    position and rotation errors.
    """
    letters = ["I", "C", "R", "A"]
    fig, axes = plt.subplots(4, 1, figsize=(6, 16), sharex=True)

    for idx, letter in enumerate(letters):
        npz_path = os.path.join(
            "results", "traj_following", f"get_icra_traj_{letter}", "get_icra_traj.npz"
        )
        current_ax = axes[idx] if isinstance(axes, np.ndarray) else axes
        visualize_error(npz_path, current_ax)
        # Left-side label for each row for readability
        set_2d_tick_labelsize(current_ax)
        # Right y-axis label already set by visualize_error via twinx

    set_2d_tick_labelsize(axes[-1])

    fig.tight_layout()
    save_path = os.path.join("results", "error_icra.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=900)
    print(f"Saved error ICRA plot to {save_path}")


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
    # plot_constrain_motion_planning()
    plot_tendon_length()
    # plot_error()
