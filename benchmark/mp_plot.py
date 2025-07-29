import numpy as np
import matplotlib.pyplot as plt
from soul.visualization.visualizer_plot import visualize_mp_scene


def plot_mp_with_coll_scene():
    fig = plt.figure(facecolor="white", figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection="3d")

    save_path = "results/5.water_flowers/mp_with_coll_sections_3_eval_100.npz"
    world_config_path = "configs/maps/mp_scene/obstacles_5.water_flowers.json"
    selected_indices = [18]

    try:
        data = np.load(save_path)
    except FileNotFoundError:
        print(f"Warning: Data file not found at {save_path}. Skipping plot.")
        ax1.text(
            0.5,
            0.5,
            0.5,
            "Data file not found",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax1.transAxes,
        )
        plt.tight_layout()
        plt.savefig("results/mp_scene_examples.png")
        plt.close()
        return

    all_target_positions = data["target_position"]
    target_wxyz = data["target_wxyz"]
    fk_result = data["fk_result"]
    start_fk_result = data["start_fk_result"]
    planned_tip_traj = data["planned_tip_traj"]

    # Visualize all sampled target points to check distribution
    ax1.scatter(
        all_target_positions[:, 0],
        all_target_positions[:, 1],
        all_target_positions[:, 2],
        c="silver",
        marker="x",
        s=15,
        label="All Sampled Targets",
    )

    # Select some solutions to visualize in detail
    num_solutions = len(fk_result)
    if selected_indices is None:
        num_to_show = min(3, num_solutions)
        indices_to_vis = np.random.choice(
            num_solutions, size=num_to_show, replace=False
        )
    else:
        indices_to_vis = selected_indices

    indices_to_vis = [idx for idx in indices_to_vis if idx < num_solutions]

    print(f"Visualizing IK solutions for indices: {indices_to_vis}")

    selected_fk_result = fk_result[indices_to_vis]
    selected_target_position = all_target_positions[indices_to_vis]
    selected_target_wxyz = target_wxyz[indices_to_vis]
    selected_start_fk_result = start_fk_result[indices_to_vis]
    selected_planned_tip_traj = planned_tip_traj[indices_to_vis]
    start_tip_positions = selected_start_fk_result[:, -1, :3, 3]

    visualize_mp_scene(
        pose=selected_fk_result,
        initial_pose=selected_start_fk_result,  # Show start and end robot poses
        num_points=10,
        target_wxyz=selected_target_wxyz,
        target_position=selected_target_position,
        world_coll_config=world_config_path,
        save_path=save_path.replace(".npz", "_fk_scene.png"),
        ax=ax1,
    )

    # Draw lines and trajectories for each selected index
    for i in range(len(indices_to_vis)):
        start_pos = start_tip_positions[i]
        end_pos = selected_target_position[i]
        # Draw a line from start to end tip position
        ax1.plot(
            [start_pos[0], end_pos[0]],
            [start_pos[1], end_pos[1]],
            [start_pos[2], end_pos[2]],
            "k--",
            label=f"Ideal Path {indices_to_vis[i]}" if i == 0 else "",
        )

    ax1.text2D(
        0.05,
        0.95,
        "Green: Start Pose, Blue: Solved Pose, Red: Target Pose",
        transform=ax1.transAxes,
    )

    ax1.legend()
    plt.tight_layout()
    plt.savefig("results/mp_scene_examples.png")
    plt.close()


if __name__ == "__main__":
    plot_mp_with_coll_scene()
