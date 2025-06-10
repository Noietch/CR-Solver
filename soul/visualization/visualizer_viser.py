import viser
import numpy as np
import jax.numpy as jnp
import jaxlie
from ..collision.pcc_robot_collision import RobotCollision
from ..collision.geometry import Sphere, Capsule
from ..robots.pcc_robot import ConstantCurvatureState, PCCRobot


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

        # Create initial spheres with default positions (will be updated later)
        self._create_sphere_visualizations()

    def _create_sphere_visualizations(self):
        """Create sphere visualizations for the robot collision model."""
        # Clear any existing sphere handles
        # for handle in self.sphere_handles:
        #     handle.remove()
        self.sphere_handles = []

        # Assume spheres have a consistent radius across the robot
        # This is a reasonable assumption based on how RobotCollision is initialized
        if isinstance(self.robot_coll.coll, Sphere):
            num_spheres = self.robot_coll.coll.radius.shape[0]
            # breakpoint()
            # Create a mesh for each sphere in the model
            for i in range(num_spheres):
                sphere_node_name = f"{self.root_node_name}/sphere_{i}"
                sphere_handle = self.server.scene.add_mesh_trimesh(
                    name=sphere_node_name,
                    mesh=Sphere.from_center_and_radius(
                        np.array([0, 0, 0]), np.array([self.robot_coll.coll.radius[i]])
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
            # wxyz = se3.rotation().wxyz
            handle.position = np.array(position)
            # handle.wxyz = np.array(wxyz)

    def visualize_traj_collisions(self, robot: PCCRobot, cfg: ConstantCurvatureState):
        """Visualize a capsule."""
        for i in range(len(cfg) - 1):
            swept_capsules = self.robot_coll.get_swept_capsules(
                robot, cfg[i], cfg[i + 1]
            )
            sphere_handle = self.server.scene.add_mesh_trimesh(
                name=f"{self.root_node_name}_traj_collisions/swept_capsule_{i}",
                mesh=swept_capsules.to_trimesh(),
            )
            self.sphere_handles.append(sphere_handle)
