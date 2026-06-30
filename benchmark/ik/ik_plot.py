import matplotlib.pyplot as plt
import numpy as np

from cr_solver.visualization.visualizer_plot import visualize_cc_model_3d


def visualize_ik_with_coll(
    save_path: str,
    world_config_path: str,
    ax: plt.Axes,
    selected_indices: list[int],
    x_limit: tuple[float, float] = (-1.3, 1.3),
    y_limit: tuple[float, float] = (-1.3, 1.3),
    z_limit: tuple[float, float] = (-1.3, 1.3),
):
    data = np.load(save_path)
    target_position = data["target_position"]
    target_wxyz = data["target_wxyz"]
    fk_result = data["fk_result"]
    # Randomly select 3 solutions
    num_solutions = len(fk_result)
    if selected_indices is None:
        selected_indices = np.random.choice(
            num_solutions, size=3, replace=False
        )
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
        color="black",
        x_limit=x_limit,
        y_limit=y_limit,
        z_limit=z_limit,
    )


def plot_ik_with_coll():
    fig = plt.figure(facecolor="white", figsize=(8, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    visualize_ik_with_coll(
        "results/ik_with_coll_lattice/ik_with_coll_sections_3_eval_100.npz",
        "configs/maps/ik_maps/obstacles_lattice.json",
        ax=ax1,
        selected_indices=[0, 63, 97],
        y_limit=(-0.8, 1.8),
    )
    visualize_ik_with_coll(
        "results/ik_with_coll_icosahedron/"
        "ik_with_coll_sections_6_eval_100.npz",
        "configs/maps/ik_maps/obstacles_icosahedron.json",
        ax=ax2,
        selected_indices=[83, 32, 90],
    )
    plt.tight_layout()
    plt.savefig("results/ik_examples.png")
    plt.close()


if __name__ == "__main__":
    plot_ik_with_coll()
