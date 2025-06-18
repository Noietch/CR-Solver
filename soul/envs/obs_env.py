import jax
import jax.numpy as jnp
import json
import jaxlie
import trimesh
import viser
import time

from ..robots.pcc_robot import PCCRobot, ConstantCurvatureState
from ..collision import RobotCollision
from ..collision import CollGeom, Sphere, Capsule, HalfSpace
from ..visualization.visualizer_plot import visualize_pcc_model_2d


class ObstacleEnv:
    def __init__(self, map_config: str | dict):
        if isinstance(map_config, str):
            with open(map_config, "r") as f:
                self.map_config = json.load(f)
        else:
            self.map_config = map_config
        self.obstacle_list = self._load_map()

        self.obstacle_position_handle_dict = {}
        self.obstacle_handle_dict = {}

    def _load_map(self, viser=True):
        obstacle_list: list[tuple[str, CollGeom]] = []
        for name, value in self.map_config.items():
            if value["type"] == "sphere":
                sphere = Sphere.from_center_and_radius(
                    center=(
                        jnp.array([0.0, 0.0, 0.0])
                        if viser
                        else jnp.array(value["position"])
                    ),
                    radius=jnp.array(value["radius"]),
                )
                obstacle_list.append((name, sphere))
            elif value["type"] == "capsule":
                capsule = Capsule.from_radius_height(
                    radius=jnp.array(value["radius"]),
                    height=jnp.array(value["height"]),
                    position=(
                        jnp.array([0.0, 0.0, 0.0])
                        if viser
                        else jnp.array(value["position"])
                    ),
                    wxyz=jnp.array(value["wxyz"]),
                )
                obstacle_list.append((name, capsule))
        return obstacle_list

    def get_collision_list(self):
        obstacle_list = self._load_map(viser=False)
        collision_list = []
        for _, obstacle in obstacle_list:
            collision_list.append(obstacle)
        plane_coll = HalfSpace.from_point_and_normal(
            jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
        )
        collision_list.append(plane_coll)
        return collision_list

    def to_merged_trimesh(self):
        # merge all obstacles into one trimesh
        trimesh_list = []
        for obstacle in self.obstacle_list:
            trimesh_list.append(obstacle.to_trimesh())
        return trimesh.util.concatenate(trimesh_list)

    def visualize(
        self,
        server: viser.ViserServer,
        editor: bool = False,
        export_path: str = None,
    ):
        # create obstacle position handles
        if editor:
            for name, obstacle in self.obstacle_list:
                obstacle_handle = server.scene.add_transform_controls(
                    f"/obstacle/{name}",
                    scale=0.8,
                    position=jnp.array(self.map_config[name]["position"]),
                )
                obstacle_position_handle = server.gui.add_vector3(
                    f"{name} Position",
                    initial_value=jnp.array(self.map_config[name]["position"]),
                    step=0.01,
                    min=None,
                    max=None,
                )

                # Create a closure to capture the current values for GUI -> Transform control
                def create_gui_update_callback(pos_handle, obs_handle):
                    def update_callback(args):
                        obs_handle.position = pos_handle.value

                    return update_callback

                obstacle_position_handle.on_update(
                    create_gui_update_callback(
                        obstacle_position_handle, obstacle_handle
                    )
                )
                self.obstacle_position_handle_dict[name] = obstacle_position_handle
                self.obstacle_handle_dict[name] = obstacle_handle

        # add obstacles to the scenes
        for name, obstacle in self.obstacle_list:
            server.scene.add_mesh_trimesh(
                f"/obstacle/{name}/mesh", mesh=obstacle.to_trimesh()
            )
        server.scene.add_grid("/ground", width=6, height=6)

        # add a button to reset the obstacle positions
        export_button = server.gui.add_button(
            "Export Obstacle Positions", icon="refresh"
        )

        def export_obstacle_positions(args):
            if export_path is None:
                print("No export path provided")
                return
            output_dict = {}
            for name, obstacle in self.obstacle_list:
                if isinstance(obstacle, Sphere):
                    output_dict[name] = {
                        "type": "sphere",
                        "position": list(self.obstacle_handle_dict[name].position),
                        "radius": float(obstacle.radius),
                    }
                elif isinstance(obstacle, Capsule):
                    output_dict[name] = {
                        "type": "capsule",
                        "position": list(self.obstacle_handle_dict[name].position),
                        "radius": float(obstacle.radius),
                        "height": float(obstacle.height),
                        "wxyz": list(self.obstacle_handle_dict[name].wxyz),
                    }
                else:
                    raise ValueError(f"Unsupported obstacle type: {type(obstacle)}")
            with open(export_path, "w") as f:
                json.dump(output_dict, f, indent=2)
            print(f"Obstacle positions exported to {export_path}")

        export_button.on_click(export_obstacle_positions)

    def show(self, editor: bool = False, export_path: str = None):
        server = viser.ViserServer(port=8081)
        self.visualize(server, editor, export_path)
        # update obstacle position handles (Transform control -> GUI)
        while True:
            for name, obstacle_handle in self.obstacle_handle_dict.items():
                self.obstacle_position_handle_dict[name].value = (
                    obstacle_handle.position
                )
            time.sleep(0.01)
