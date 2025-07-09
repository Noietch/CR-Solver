import viser
import numpy as np
import jax.numpy as jnp
import jaxlie
import jax_dataclasses as jdc
from ..geom.collision_cc_robot import RobotCollision
from ..geom.collision_world import WorldCollision
from ..geom.geometry import Sphere
from ..robots.cc_robot import ConstantCurvatureState, CCRobot
from functools import partial
import jax
import json

class ViserSoftRobot:
    def __init__(
        self,
        server: viser.ViserServer | viser.ClientHandle,
        robot_coll: RobotCollision,
        root_node_name: str,
    ):
        self.server = server
        self.robot_coll = robot_coll
        self.root_node_name = root_node_name
        self.sphere_handles = []

    def create_sphere_visualizations(self):
        """Create sphere visualizations for the robot collision model."""
        # Clear any existing sphere handles
        # for handle in self.sphere_handles:
        #     handle.remove()
        self.sphere_handles = []

        # Assume spheres have a consistent radius across the robot
        # This is a reasonable assumption based on how RobotCollision is initialized
        if isinstance(self.robot_coll.coll, Sphere):
            num_spheres = self.robot_coll.coll.radius.shape[0]
            # Create a mesh for each sphere in the model
            for i in range(num_spheres):
                sphere_node_name = f"{self.root_node_name}/sphere_{i}"
                sphere_handle = self.server.scene.add_mesh_trimesh(
                    name=sphere_node_name,
                    mesh=Sphere.from_center_and_radius(
                        jnp.array([0, 0, 0]), jnp.array([self.robot_coll.coll.radius[i]])
                    ).to_trimesh(),
                )
                self.sphere_handles.append(sphere_handle)

    def update_cfg(self, all_poses: jnp.ndarray):
        """Update visualization with new robot configuration poses."""
        # This method is called with the output of robot.forward_kinematics()
        self.update_pose(all_poses)

    def update_pose(self, all_poses: jnp.ndarray):
        """Update visualization with new poses."""
        # Convert the robot poses to sphere positions
        # First, get the collision geometry at the current pose
        for handle, pose in zip(self.sphere_handles, all_poses):
            se3 = jaxlie.SE3.from_matrix(pose)
            position = se3.translation()
            handle.position = np.array(position)

    def visualize_traj_collisions(self, robot: CCRobot, cfg: ConstantCurvatureState):
        """Visualize a capsule."""
        traj_len = cfg.theta.shape[0]
        for i in range(traj_len - 1):
            cfg_i = jax.tree.map(lambda x: x[i], cfg)
            cfg_i_plus_1 = jax.tree.map(lambda x: x[i + 1], cfg)
            swept_capsules = self.robot_coll.get_swept_capsules(
                robot, cfg_i, cfg_i_plus_1
            )
            self.server.scene.add_mesh_trimesh(
                name=f"{self.root_node_name}_traj_collisions/swept_capsule_{i}",
                mesh=swept_capsules.to_trimesh(),
            )

    def visualize_traj(
        self,
        traj: jnp.ndarray | jaxlie.SE3,
        color: np.ndarray = np.array([1.0, 0.0, 0.0]),
        name: str = "traj",
    ):
        if isinstance(traj, jaxlie.SE3):
            traj = traj.translation()
        else:
            traj = jaxlie.SE3.from_matrix(traj).wxyz_xyz[..., -1, 4:]
        self.server.scene.add_point_cloud(
            name=f"{self.root_node_name}_{name}",
            points=np.array(traj),
            colors=color,
            point_size=0.01,
            point_shape="circle",
        )


class ViserWorld:
    def __init__(
        self,
        server: viser.ViserServer | viser.ClientHandle,
        world_coll: WorldCollision,
        is_handle_able: bool = False,
        config_path: str | None = None,
    ):
        self.server = server
        self.world_coll = world_coll
        self.is_handle_able = is_handle_able
        self.config_path = config_path
        if self.is_handle_able:
            self.save_button = self.server.add_gui_button("Save Poses")
            if self.config_path is not None:
                self.save_path_handle = self.server.add_gui_text("Save Path", initial_value=self.config_path)
            else:
                self.save_path_handle = self.server.add_gui_text("Save Path", initial_value="configs/maps/obstacles_new.json")

            @self.save_button.on_click
            def _(_):
                self.save_obstacle_poses(self.save_path_handle.value)
                self.save_button.disabled = True
        
    def update_obstacle_pose(self, index, event):
        new_pose_slice = np.concatenate([event.wxyz, event.position])
        new_poses = self.world_coll.obstacles.pose.wxyz_xyz.at[index].set(
            new_pose_slice
        )
        
        # Create a new pose object
        new_pose = jaxlie.SE3(new_poses)
        
        # Create a new obstacles object with the updated pose
        new_obstacles = jdc.replace(self.world_coll.obstacles, pose=new_pose)

        # Create a new world_coll object with the updated obstacles
        self.world_coll = jdc.replace(self.world_coll, obstacles=new_obstacles)
        if self.save_button.disabled:
            self.save_button.disabled = False

    def save_obstacle_poses(self, path: str):
        """Saves the current obstacle poses to a JSON file."""
        # Assuming obstacles are spheres, which is consistent with the config format
        if not isinstance(self.world_coll.obstacles, Sphere):
            print("Saving only supported for Sphere obstacles.")
            return

        obstacles_dict = {}
        centers = self.world_coll.obstacles.pose.translation()
        radii = self.world_coll.obstacles.radius
        for i in range(len(centers)):
            obstacles_dict[f"obstacle_{i+1}"] = {
                "type": "sphere",
                "center": [round(float(center), 2) for center in centers[i]],
                "radius": round(float(radii[i]), 2),
            }

        with open(path, "w") as f:
            json.dump(obstacles_dict, f, indent=4)
        
        print(f"Obstacle poses saved to {path}")

    def create_mesh_visualizations(self):
        """Create mesh visualizations for the obstacles."""
        # add mesh visualizations
        if self.is_handle_able:
            for i, mesh in enumerate(self.world_coll.mesh):
                obstacle_pose = self.world_coll.obstacles.pose.wxyz_xyz[i]
                obstacle_handle = self.server.scene.add_transform_controls(
                    f"obstacles/handle_{i}",
                    scale=0.5,
                    position=np.array(obstacle_pose[4:]),
                    wxyz=np.array(obstacle_pose[:4]),
                )
                mesh.apply_translation(-obstacle_pose[4:])
                # jax.debug.breakpoint()
                self.server.scene.add_mesh_trimesh(
                    name=f"obstacles/handle_{i}/mesh",
                    mesh=mesh,
                )

                obstacle_handle.on_update(partial(self.update_obstacle_pose, i))

        else:
            self.server.scene.add_mesh_trimesh(
                name=f"obstacles/collision",
                mesh=self.world_coll.obstacles.to_trimesh(),
            )

        # add ground visualizations
        self.server.scene.add_grid("/ground", width=6, height=6)
