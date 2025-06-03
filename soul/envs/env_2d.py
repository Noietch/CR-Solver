import jax
import jax.numpy as jnp
import json
import jaxlie

from ..robots.pcc_robot import PCCRobot, ConstantCurvatureState
from ..collision import RobotCollision
from ..collision import HalfSpace, Sphere
from ..visualization.visualizer_plot import visualize_pcc_model_2d


class Env2D:
    def __init__(self, map_config: str, robot_config: str):
        self.map_config = map_config
        self.robot_config = robot_config
        self.robot_pose = None

    def _load_robot(self):
        # load robot config
        with open(self.robot_config, "r") as f:
            self.robot_config = json.load(f)
        self.robot_all_length = self.robot_config["length"] * self.robot_config["num_sections"]
        self.num_points = self.robot_config["num_points_per_section"]
        robot_base_position = jnp.array([0, 0, -self.robot_all_length])
        # create robot and robot collision
        self.robot = PCCRobot.from_config(self.robot_config, default_base_position=robot_base_position)
        # create robot collision
        self.robot_coll = RobotCollision.from_config(self.robot_config)

    @staticmethod
    def _convert_map_coord(coord: list):
        return jnp.array([coord[0], 0, coord[1]])

    @staticmethod
    def _convert_obstacle_coord(coord: list):
        # convert to obstacles sphere
        return jnp.array([coord[0], 0, coord[1], coord[3]])

    def _load_map(self):
        with open(self.map_config, "r") as f:
            self.map = json.load(f)
        # convert map coord to world coord
        self.goal = self._convert_map_coord(self.map["target_position"])
        self.obstacles = jnp.stack([self._convert_obstacle_coord(ob) for ob in self.map["obstacle_sphere"]])
    
    def reset(self):
        self._load_map()
        self._load_robot()

    def _get_observation(self):
        tip_transform = self.robot_pose[-1, ...]
        tip_pose = jaxlie.SE3.from_matrix(tip_transform)
        # Select the obstacles that are in collision with the tip
        tip_pos = tip_pose.translation()
        # obstacles: shape (num_obstacles, 4) where [x, y, z, r]
        obs_centers = self.obstacles[:, :3]
        obs_radii = self.obstacles[:, 3]
        dists = jnp.linalg.norm(obs_centers - tip_pos, axis=-1)
        in_collision = dists <= obs_radii
        collided_obstacles = self.obstacles[in_collision]
        return collided_obstacles

    def step(self, cfg: ConstantCurvatureState):
        self.robot_pose = self.robot.forward_kinematics(cfg)
        return self._get_observation()

    def render(self, mode="plot"):
        if mode == "plot":
            self.plot()
        elif mode == "viser":
            self.viser()
        else:
            raise ValueError(f"Invalid render mode: {mode}")

    def plot(self, save_path: str = './visualization'):
        if self.robot_pose is None:
            raise ValueError("Robot pose is not set")
        visualize_pcc_model_2d(self.robot_pose, self.goal, num_points=self.num_points, obstacles=self.obstacles, save_path=save_path)