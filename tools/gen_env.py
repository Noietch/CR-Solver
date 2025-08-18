import json
import os
import numpy as np

from soul.visualization.visualizer_plot import visualize_cc_model_3d


def generate_lattice_env(save_path: str, radius: float = 0.2):
    """
    Generate a 3D lattice environment with spherical obstacles.

    Parameters:
    - save_path: Path to save the generated obstacle configuration JSON file
    - radius: Radius of the spherical obstacles

    The function creates spherical obstacles arranged in a square lattice:
    - Lattice spacing: 0.8 in x and y axes, 1.0 in z axis
    - Creates a 3x3x3 grid centered around origin
    """
    # Parameters for the lattice
    x_spacing = 0.8
    y_spacing = 0.8
    z_spacing = 1.0

    # Define the grid range (3x3x3 grid centered at origin)
    x_positions = np.array([-1.0, 0.0, 1.0]) * x_spacing
    y_positions = np.array([-1.0, 0.0, 1.0]) * y_spacing
    z_positions = np.array([-0.8, -0.8 + z_spacing, -0.8 + 2 * z_spacing])

    obstacles_dict = {}
    obstacle_count = 1

    # Generate obstacles in lattice pattern
    for x in x_positions:
        for y in y_positions:
            for z in z_positions:
                obstacle_key = f"obstacle_{obstacle_count}"
                obstacles_dict[obstacle_key] = {
                    "type": "sphere",
                    "center": [float(x), float(y), float(z)],
                    "radius": radius,
                }
                obstacle_count += 1

    # Save to JSON file
    with open(save_path, "w") as f:
        json.dump(obstacles_dict, f, indent=4)

    print(
        f"Generated lattice environment with {len(obstacles_dict)} obstacles saved to {save_path}"
    )


def generate_octahedron_env(
    save_path: str,
    center_pos: list = [0.0, 0.0, 0.0],
    scale: float = 1.0,
    radius: float = 0.15,
):
    """
    Generate an octahedron environment with spherical obstacles at vertices.

    Parameters:
    - save_path: Path to save the generated obstacle configuration JSON file
    - center_pos: Center position of the octahedron [x, y, z]
    - scale: Scale factor for the octahedron size
    - radius: Radius of the spherical obstacles

    The octahedron has 6 vertices positioned at ±scale along each axis.
    """

    # Octahedron vertices (6 vertices)
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],  # +X
            [-1.0, 0.0, 0.0],  # -X
            [0.0, 1.0, 0.0],  # +Y
            [0.0, -1.0, 0.0],  # -Y
            [0.0, 0.0, 1.0],  # +Z
            [0.0, 0.0, -1.0],  # -Z
        ]
    ) * scale + np.array(center_pos)

    obstacles_dict = {}
    for i, vertex in enumerate(vertices):
        obstacle_key = f"obstacle_{i+1}"
        obstacles_dict[obstacle_key] = {
            "type": "sphere",
            "center": [float(vertex[0]), float(vertex[1]), float(vertex[2])],
            "radius": radius,
        }

    # Save to JSON file
    with open(save_path, "w") as f:
        json.dump(obstacles_dict, f, indent=4)

    print(
        f"Generated octahedron environment with {len(obstacles_dict)} obstacles saved to {save_path}"
    )


def generate_cube_env(
    save_path: str,
    center_pos: list = [0.0, 0.0, 0.0],
    scale: float = 1.0,
    radius: float = 0.15,
):
    """
    Generate a cube environment with spherical obstacles at vertices.

    Parameters:
    - save_path: Path to save the generated obstacle configuration JSON file
    - center_pos: Center position of the cube [x, y, z]
    - scale: Scale factor for the cube size
    - radius: Radius of the spherical obstacles

    The cube has 8 vertices positioned at all combinations of ±scale.
    """

    # Cube vertices (8 vertices)
    vertices = np.array(
        [
            [1.0, 1.0, 1.0],  # +++
            [1.0, 1.0, -1.0],  # ++-
            [1.0, -1.0, 1.0],  # +-+
            [1.0, -1.0, -1.0],  # +--
            [-1.0, 1.0, 1.0],  # -++
            [-1.0, 1.0, -1.0],  # -+-
            [-1.0, -1.0, 1.0],  # --+
            [-1.0, -1.0, -1.0],  # ---
        ]
    ) * scale + np.array(center_pos)

    obstacles_dict = {}
    for i, vertex in enumerate(vertices):
        obstacle_key = f"obstacle_{i+1}"
        obstacles_dict[obstacle_key] = {
            "type": "sphere",
            "center": [float(vertex[0]), float(vertex[1]), float(vertex[2])],
            "radius": radius,
        }

    # Save to JSON file
    with open(save_path, "w") as f:
        json.dump(obstacles_dict, f, indent=4)

    print(
        f"Generated cube environment with {len(obstacles_dict)} obstacles saved to {save_path}"
    )


def generate_icosahedron_env(
    save_path: str,
    center_pos: list = [0.0, 0.0, 0.0],
    scale: float = 1.0,
    radius: float = 0.15,
):
    """
    Generate an icosahedron environment with spherical obstacles at vertices.

    Parameters:
    - save_path: Path to save the generated obstacle configuration JSON file
    - center_pos: Center position of the icosahedron [x, y, z]
    - scale: Scale factor for the icosahedron size
    - radius: Radius of the spherical obstacles

    The icosahedron has 12 vertices based on the golden ratio.
    """

    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Icosahedron vertices (12 vertices)
    # Three orthogonal golden rectangles
    vertices = np.array(
        [
            # Rectangle in XY plane
            [1.0, phi, 0.0],
            [-1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [-1.0, -phi, 0.0],
            # Rectangle in YZ plane
            [0.0, 1.0, phi],
            [0.0, -1.0, phi],
            [0.0, 1.0, -phi],
            [0.0, -1.0, -phi],
            # Rectangle in XZ plane
            [phi, 0.0, 1.0],
            [-phi, 0.0, 1.0],
            [phi, 0.0, -1.0],
            [-phi, 0.0, -1.0],
        ]
    )

    # Normalize to unit sphere and scale
    vertices = vertices / np.linalg.norm(vertices[0]) * scale + np.array(center_pos)

    obstacles_dict = {}
    for i, vertex in enumerate(vertices):
        obstacle_key = f"obstacle_{i+1}"
        obstacles_dict[obstacle_key] = {
            "type": "sphere",
            "center": [float(vertex[0]), float(vertex[1]), float(vertex[2])],
            "radius": radius,
        }

    # Save to JSON file
    with open(save_path, "w") as f:
        json.dump(obstacles_dict, f, indent=4)

    print(
        f"Generated icosahedron environment with {len(obstacles_dict)} obstacles saved to {save_path}"
    )


def generate_random_env(
    save_path: str,
    num_obstacles: int = 10,
    center_pos: list = [0.0, 0.0, 0.0],
    scale: float = 1.0,
    radius_min: float = 0.15,
    radius_max: float = 0.30,
    robot_radius: float = 0.1,
    start_from_initialization: bool = False,
):
    """
    Generate a random environment with spherical obstacles.

    Parameters:
    - save_path: Path to save the generated obstacle configuration JSON file
    - num_obstacles: Number of spherical obstacles to generate
    - center_pos: Center position of the environment [x, y, z]
    - scale: Scale factor for the environment size
    - radius: Radius of the spherical obstacles
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    obstacles_dict = {}

    def generate_one_obstacle(i: int) -> dict:
        obstacle_key = f"obstacle_{i+1}"
        center = [
            np.random.uniform(-scale, scale) + center_pos[0],
            np.random.uniform(-scale, scale) + center_pos[1],
            np.random.uniform(-scale, scale) + center_pos[2],
        ]
        radius = np.random.uniform(radius_min, radius_max)
        obstacles_dict[obstacle_key] = {
            "type": "sphere",
            "center": center,
            "radius": radius,
        }
        return obstacles_dict

    def whether_obstacle_in_collision(x, y) -> bool:
        # Check if the obstacle is within the robot's radius
        obstacle_center = np.array([x, y, 0])
        return np.linalg.norm(obstacle_center) < robot_radius

    for i in range(num_obstacles):
        obstacle_dict = generate_one_obstacle(i)
        if start_from_initialization:
            x, y = (
                obstacle_dict[f"obstacle_{i+1}"]["center"][0],
                obstacle_dict[f"obstacle_{i+1}"]["center"][1],
            )
            is_in_base_point = whether_obstacle_in_collision(x, y)
            while is_in_base_point:
                obstacle_dict = generate_one_obstacle(i)
                x, y = (
                    obstacle_dict[f"obstacle_{i+1}"]["center"][0],
                    obstacle_dict[f"obstacle_{i+1}"]["center"][1],
                )
                is_in_base_point = whether_obstacle_in_collision(x, y)
        obstacles_dict.update(obstacle_dict)

    # Save to JSON file
    with open(save_path, "w") as f:
        json.dump(obstacles_dict, f, indent=4)

    print(
        f"Generated random environment with {len(obstacles_dict)} obstacles saved to {save_path}"
    )


if __name__ == "__main__":
    # # Lattice (27 vertices)
    # generate_lattice_env("configs/maps/ik_maps/obstacles_lattice.json", radius=0.2)
    # visualize_cc_model_3d(
    #     world_coll_config="configs/maps/ik_maps/obstacles_lattice.json",
    #     save_path="visualization/lattice_env.png",
    # )

    # # Octahedron (6 vertices) - smaller radius
    # generate_octahedron_env(
    #     "configs/maps/ik_maps/obstacles_octahedron.json", radius=0.4, scale=0.8
    # )
    # visualize_cc_model_3d(
    #     world_coll_config="configs/maps/ik_maps/obstacles_octahedron.json",
    #     save_path="visualization/octahedron_env.png",
    # )

    # # Cube (8 vertices) - medium radius
    # generate_cube_env("configs/maps/ik_maps/obstacles_cube.json", radius=0.4, scale=0.8)
    # visualize_cc_model_3d(
    #     world_coll_config="configs/maps/ik_maps/obstacles_cube.json",
    #     save_path="visualization/cube_env.png",
    # )

    # # Icosahedron (12 vertices) - larger radius
    # generate_icosahedron_env(
    #     "configs/maps/ik_maps/obstacles_icosahedron.json", radius=0.4, scale=1.0
    # )
    # visualize_cc_model_3d(
    #     world_coll_config="configs/maps/ik_maps/obstacles_icosahedron.json",
    #     save_path="visualization/icosahedron_env.png",
    # )

    # Random environment
    section_list = [3]
    section_length = 1.0
    start_from_initialization = True

    for section in section_list:
        generate_random_env(
            f"configs/maps/mp_scene/obstacles_random_start_init_{start_from_initialization}_section_{section}.json",
            num_obstacles=section**3,  # Number of obstacles increases with section size
            center_pos=[0.0, 0.0, 0.0],
            scale=section * section_length,
            radius_min=0.1,
            radius_max=0.2,
            robot_radius=0.01,
            start_from_initialization=start_from_initialization,
        )
        visualize_cc_model_3d(
            world_coll_config=f"configs/maps/mp_scene/obstacles_random_start_init_{start_from_initialization}_section_{section}.json",
            save_path=f"visualization/obstacles_random_start_init_{start_from_initialization}_section_{section}.png",
        )

    print("\nAll environments generated successfully!")
