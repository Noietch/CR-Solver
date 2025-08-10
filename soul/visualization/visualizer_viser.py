import jax
import json
import viser
import trimesh
import numpy as np
import jax.numpy as jnp
import jaxlie
import jax_dataclasses as jdc
from functools import partial
import time
import imageio.v3 as iio
from tqdm import tqdm

from ..geom.collision_cc_robot import RobotCollision
from ..geom.collision_world import WorldCollision
from ..geom.geometry import Sphere, BoundingBox
from ..robots.cc_robot import ConstantCurvatureState, CCRobot


class ViserSoftRobot:
    def __init__(
        self,
        server: viser.ViserServer | viser.ClientHandle,
        robot: CCRobot,
        robot_coll: RobotCollision,
        root_node_name: str,
        enable_backbone: bool = True,
    ):
        self.server = server
        self.robot = robot  # 保存robot实例
        self.robot_coll = robot_coll
        self.root_node_name = root_node_name
        self.robot_config = robot.config
        self.robot_cylinder_handles = []
        self.robot_backbone_cylinder_handles = []
        self.enable_backbone = enable_backbone
        self._update_counter = 0  # For frame skipping optimization

    def reset_pose(self):
        default_state = ConstantCurvatureState(
            base_position=jnp.array([0.0, 0.0, 0.0]),
            theta=jnp.zeros(self.robot_config.num_sections) + 1e-6,
            phi=jnp.zeros(self.robot_config.num_sections) + 1e-6,
        )
        poses = self.robot.forward_kinematics(default_state)
        return poses

    def _generate_section_colors(self, num_sections):
        colors = []
        for i in range(num_sections):
            # use HSV to generate uniform colors
            hue = i / num_sections
            # convert to RGB (simplified HSV to RGB conversion, S=1, V=1)
            c = 1.0
            x = c * (1 - abs((hue * 6) % 2 - 1))
            m = 0

            if 0 <= hue < 1 / 6:
                r, g, b = c, x, 0
            elif 1 / 6 <= hue < 2 / 6:
                r, g, b = x, c, 0
            elif 2 / 6 <= hue < 3 / 6:
                r, g, b = 0, c, x
            elif 3 / 6 <= hue < 4 / 6:
                r, g, b = 0, x, c
            elif 4 / 6 <= hue < 5 / 6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors.append((r + m, g + m, b + m))
        return colors

    def _create_cylinder_mesh(
        self, radius: float, sections: int, color: tuple, alpha: float = 1.0
    ):
        """Create a cylinder mesh with specified radius, sections and color."""
        mesh = trimesh.creation.cylinder(
            radius=radius,
            height=1.0,  # unit length, adjusted later through transformation
            sections=sections,
        )

        # Apply color with alpha channel using standard RGBA format
        # Trimesh expects RGBA values in range 0-255 as uint8
        rgba_color = np.array(
            [int(c * 255) for c in color] + [int(alpha * 255)], dtype=np.uint8
        )

        # Set both vertex and face colors for maximum compatibility
        # Some renderers use vertex colors, others use face colors
        n_vertices = len(mesh.vertices)
        n_faces = len(mesh.faces)

        mesh.visual.vertex_colors = np.tile(rgba_color, (n_vertices, 1))
        mesh.visual.face_colors = np.tile(rgba_color, (n_faces, 1))

        return mesh

    def _get_section_index(self, point_index: int):
        """Get the section index for a given point index."""
        current_point_section = point_index // self.robot_config.num_points_per_section
        next_point_section = (
            point_index + 1
        ) // self.robot_config.num_points_per_section

        section_idx = max(current_point_section, next_point_section)
        section_idx = min(
            section_idx, self.robot_config.num_sections - 1
        )  # prevent out of bounds
        return section_idx

    def create_robot_visualizations(self, alpha: float = 1.0):
        # clear existing cylinders
        for handle in self.robot_cylinder_handles:
            handle.remove()
        self.robot_cylinder_handles = []

        # clear existing backbone cylinders
        for handle in self.robot_backbone_cylinder_handles:
            handle.remove()
        self.robot_backbone_cylinder_handles = []

        # calculate total number of points
        total_points = (
            self.robot_config.num_sections * self.robot_config.num_points_per_section
        )
        cylinder_radius = self.robot_config.radius * 0.8
        backbone_radius = self.robot_config.radius * 0.1  # smaller radius for backbone

        # generate colors for each section
        section_colors = self._generate_section_colors(self.robot_config.num_sections)
        black_color = (0.0, 0.0, 0.0)  # black color for backbone

        # create n-1 cylinders connecting n nodes
        for i in range(total_points - 1):
            section_idx = self._get_section_index(i)

            # create robot body cylinder
            color = section_colors[section_idx]
            cylinder_mesh = self._create_cylinder_mesh(
                cylinder_radius, 16, color, alpha
            )
            cylinder_handle = self.server.scene.add_mesh_trimesh(
                name=f"{self.root_node_name}/cylinder_{i}",
                mesh=cylinder_mesh,
            )
            self.robot_cylinder_handles.append(cylinder_handle)

            # create backbone cylinder (black) - only if enabled
            if self.enable_backbone:
                backbone_mesh = self._create_cylinder_mesh(
                    backbone_radius, 8, black_color, alpha
                )
                backbone_handle = self.server.scene.add_mesh_trimesh(
                    name=f"{self.root_node_name}/backbone_{i}",
                    mesh=backbone_mesh,
                )
                self.robot_backbone_cylinder_handles.append(backbone_handle)
        # set the initial pose of the robot
        self.update_pose(self.reset_pose())

    def _update_cylinder_pose(
        self, handle, all_poses, i, scale_factor=0.6, cylinder_type="cylinder"
    ):
        """Update a single cylinder's position, rotation and scale."""
        # get the positions of the two adjacent nodes - convert to numpy immediately
        pos1 = np.array(jaxlie.SE3.from_matrix(all_poses[i]).translation())
        pos2 = np.array(jaxlie.SE3.from_matrix(all_poses[i + 1]).translation())

        # calculate the center position and direction of the cylinder
        center = (pos1 + pos2) / 2.0
        direction = pos2 - pos1
        length = np.linalg.norm(direction)

        if length > 1e-6:
            # calculate the rotation from the Z-axis to the target direction
            z_axis = np.array([0.0, 0.0, 1.0])
            direction_unit = direction / length
            rotation = self._compute_rotation_z_to_direction(z_axis, direction_unit)

            # check if the rotation is valid
            if np.any(np.isnan(rotation.wxyz)):
                print(f"Warning: NaN rotation for {cylinder_type} {i}")
                rotation = jaxlie.SO3.identity()

            # update the position, direction and length of the cylinder
            handle.position = tuple(center)
            handle.wxyz = tuple(rotation.wxyz)
            handle.scale = (1.0, 1.0, float(length) * scale_factor)
        else:
            # the length is too small, hide the cylinder
            handle.scale = (0.0, 0.0, 0.0)

    def update_pose(self, all_poses, skip_frames=0):
        """update the position and direction of the cylinders based on the robot state.

        Args:
            all_poses: SE3 pose matrices of all robot nodes, shape (n_points, 4, 4)
            skip_frames: Skip every N frames for performance (0 = update every frame)
        """
        if not self.robot_cylinder_handles:
            return

        # Frame skipping for performance optimization
        if skip_frames > 0:
            self._update_counter += 1
            if self._update_counter % (skip_frames + 1) != 0:
                return

        # Convert JAX arrays to numpy once at the beginning to avoid repeated conversions
        if hasattr(all_poses, "device"):  # Check if it's a JAX array
            all_poses = np.array(all_poses)

        # update main robot cylinders
        for i, handle in enumerate(self.robot_cylinder_handles):
            self._update_cylinder_pose(
                handle, all_poses, i, scale_factor=0.6, cylinder_type="cylinder"
            )

        # update backbone cylinders with the same poses - only if enabled and exist
        if self.enable_backbone and self.robot_backbone_cylinder_handles:
            for i, backbone_handle in enumerate(self.robot_backbone_cylinder_handles):
                self._update_cylinder_pose(
                    backbone_handle,
                    all_poses,
                    i,
                    scale_factor=1.0,
                    cylinder_type="backbone",
                )

    def _compute_rotation_z_to_direction(self, z_axis, target_direction):
        """calculate the rotation quaternion from the Z-axis to the target direction."""
        # check if they are parallel
        dot_product = np.dot(z_axis, target_direction)
        if np.abs(dot_product) > 0.999:
            # parallel or anti-parallel
            if dot_product > 0:
                return jaxlie.SO3.identity()
            else:
                # 180 degree rotation, choose X-axis as the rotation axis
                return jaxlie.SO3.from_matrix(np.diag([1, -1, -1]))

        # calculate the rotation axis and angle
        rotation_axis = np.cross(z_axis, target_direction)
        axis_norm = np.linalg.norm(rotation_axis)

        # prevent division by zero
        if axis_norm < 1e-8:
            return jaxlie.SO3.identity()

        rotation_axis = rotation_axis / axis_norm
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

        # create rotation using axis-angle representation
        return jaxlie.SO3.from_quaternion_xyzw(
            np.concatenate([rotation_axis * np.sin(angle / 2), [np.cos(angle / 2)]])
        )

    def create_collision_visualizations(self):
        """create sphere visualizations for the robot collision model."""
        if not hasattr(self, "sphere_handles"):
            self.sphere_handles = []

        for handle in self.sphere_handles:
            handle.remove()
        self.sphere_handles = []

        # assume spheres have a consistent radius across the robot
        # this is a reasonable assumption based on how RobotCollision is initialized
        if isinstance(self.robot_coll.coll, Sphere):
            num_spheres = self.robot_coll.coll.radius.shape[0]
            # create a mesh for each sphere in the model
            for i in range(num_spheres):
                sphere_node_name = f"{self.root_node_name}/sphere_{i}"
                sphere_handle = self.server.scene.add_mesh_trimesh(
                    name=sphere_node_name,
                    mesh=Sphere.from_center_and_radius(
                        jnp.array([0, 0, 0]),
                        jnp.array([self.robot_coll.coll.radius[i]]),
                    ).to_trimesh(),
                )
                self.sphere_handles.append(sphere_handle)

    def update_collision_pose(self, all_poses: jnp.ndarray):
        """Update collision sphere positions based on robot state."""
        if not hasattr(self, "sphere_handles"):
            return

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

    def visualize_tip_traj(
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
        enable_collision: bool = True,
    ):
        self.server = server
        self.world_coll = world_coll
        self.is_handle_able = is_handle_able
        self.config_path = config_path
        self.enable_collision = enable_collision

        if self.is_handle_able:
            self.save_button = self.server.add_gui_button("Save Poses")
            if self.config_path is not None:
                self.save_path_handle = self.server.add_gui_text(
                    "Save Path",
                    initial_value=self.config_path,
                )
            else:
                self.save_path_handle = self.server.add_gui_text(
                    "Save Path",
                    initial_value="configs/maps/obstacles_generated.json",
                )

            @self.save_button.on_click
            def on_click_save_button(args):
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

        # Replace world_coll object with the updated obstacles
        self.world_coll = jdc.replace(self.world_coll, obstacles=new_obstacles)
        if self.save_button.disabled:
            self.save_button.disabled = False

    def save_obstacle_poses(self, path: str):
        """Saves the current obstacle poses to a JSON file."""
        # Assuming obstacles are spheres, which is consistent with the config format
        if isinstance(self.world_coll.obstacles, Sphere):
            obstacles_dict = {}
            centers = self.world_coll.obstacles.pose.translation()
            radii = self.world_coll.obstacles.radius
            for i in range(len(centers)):
                obstacles_dict[f"obstacle_{i+1}"] = {
                    "type": "sphere",
                    "center": [round(float(center), 4) for center in centers[i]],
                    "radius": round(float(radii[i]), 2),
                }
        elif isinstance(self.world_coll.obstacles, BoundingBox):
            obstacles_dict = {}
            centers = self.world_coll.obstacles.pose.translation()
            extents = self.world_coll.obstacles.extents
            for i in range(len(centers)):
                obstacles_dict[f"obstacle_{i+1}"] = {
                    "type": "bbox",
                    "center": [round(float(center), 4) for center in centers[i]],
                    "extents": [round(float(extent), 4) for extent in extents[i]],
                }
        else:
            raise ValueError(
                f"Unsupported obstacle type: {type(self.world_coll.obstacles)}"
            )

        with open(path, "w") as f:
            json.dump(obstacles_dict, f, indent=4)

        print(f"Obstacle poses saved to {path}")

    def create_mesh_visualizations(self):
        """Create mesh visualizations for the obstacles."""
        # add mesh visualizations
        if self.is_handle_able:
            obstacles_coll = self.world_coll.obstacles
            for i in range(len(obstacles_coll.pose.wxyz_xyz)):
                obstacle_i = jax.tree_util.tree_map(lambda x: x[i], obstacles_coll)

                obstacle_pose = obstacles_coll.pose.wxyz_xyz[i]
                obstacle_handle = self.server.scene.add_transform_controls(
                    f"obstacles/handle_{i}",
                    scale=0.5,
                    position=np.array(obstacle_pose[4:]),
                    wxyz=np.array(obstacle_pose[:4]),
                )

                mesh = self.world_coll.mesh[i]
                mesh.apply_translation(-obstacle_pose[4:])
                self.server.scene.add_mesh_trimesh(
                    name=f"obstacles/handle_{i}/mesh",
                    mesh=mesh,
                )

                if self.enable_collision:
                    coll_mesh = obstacle_i.to_trimesh()
                    coll_mesh.apply_translation(-obstacle_pose[4:])
                    collision_handle = self.server.scene.add_mesh_trimesh(
                        name=f"obstacles/handle_{i}/collision",
                        mesh=coll_mesh,
                    )

                obstacle_handle.on_update(partial(self.update_obstacle_pose, i))

        else:
            for i, mesh in enumerate(self.world_coll.mesh):
                self.server.scene.add_mesh_trimesh(
                    name=f"obstacles/mesh_{i}",
                    mesh=mesh,
                )
            if self.enable_collision:
                self.server.scene.add_mesh_trimesh(
                    name=f"obstacles/collision",
                    mesh=self.world_coll.obstacles.to_trimesh(),
                )

        # add ground visualizations
        self.server.scene.add_grid(
            "/ground",
            width=100,
            height=100,
        )


class ViserRenderer:
    def __init__(
        self,
        server: viser.ViserServer | viser.ClientHandle,
        robot: ViserSoftRobot,
        world: ViserWorld,
    ):
        self.server = server
        self.robot = robot
        self.world = world

    def render_traj_video(
        self,
        event: viser.GuiEvent,
        traj: jnp.ndarray | jaxlie.SE3,
        skip_frames: int = 0,
        save_path: str = None,
    ):
        """
        Render trajectory as a video by combining individual frames.

        Args:
            traj: Robot trajectory as either SE3 poses or numpy array
            skip_frames: Number of frames to skip for performance (0 = render every frame)
            save_path: Path to save the video file (optional)
        """
        images = []
        traj_len = traj.shape[0] if hasattr(traj, "shape") else len(traj)
        # render every frame
        for i in tqdm(range(0, traj_len, skip_frames + 1)):
            self.robot.update_pose(traj[i])
            image = event.client.get_render(height=720, width=1280)
            images.append(image)
        # save video
        if save_path:
            # Save as video file
            if save_path.endswith(".gif"):
                iio.imwrite(save_path, images, extension=".gif", loop=0)
            elif save_path.endswith(".mp4"):
                iio.imwrite(save_path, images, plugin="FFMPEG", fps=10)
            else:
                raise ValueError("Unsupported video format")
            print(f"Video saved to {save_path}")

        return images

    def render_traj_image(
        self,
        event: viser.GuiEvent,
        traj: jnp.ndarray | jaxlie.SE3,
        skip_frames: int = 0,
        save_path: str = None,
    ):
        """
        Render trajectory as a single composite image by overlaying robot poses on environment.

        Args:
            event: Viser GUI event for accessing client
            traj: Robot trajectory as either SE3 poses or numpy array
            skip_frames: Number of frames to skip for performance (0 = render every frame)
            save_path: Path to save the composite image (optional)
        """
        if save_path is None:
            raise ValueError("Save path must be provided for trajectory image")

        print("Rendering trajectory image...")

        # Step 1: Store original visibility states and render environment-only image
        self._set_robot_visibility(False)
        env_image = event.client.get_render(height=720, width=1280)

        # Step 2: Restore robot visibility and render trajectory frames
        self._set_robot_visibility(True)

        print("Rendering robot trajectory frames...")
        robot_images = self.render_traj_video(event, traj, skip_frames)

        # Start with environment as base
        env_array = np.array(env_image, dtype=np.float32)
        composite_image = env_array.copy()

        # Overlay each robot frame with alpha blending
        alpha_per_frame = 0.2  # Distribute transparency across frames

        for i, robot_frame in enumerate(robot_images):
            robot_array = np.array(robot_frame, dtype=np.float32)

            # Calculate alpha based on frame position (start/end more opaque)
            if i == 0 or i == len(robot_images) - 1:
                alpha = 0.9  # Start and end poses more opaque
            else:
                alpha = alpha_per_frame * 2  # Intermediate poses more transparent

            # Create mask for robot pixels (non-environment pixels)
            # Assuming environment pixels are relatively similar to the background
            diff = np.abs(robot_array - env_array).mean(axis=2)
            robot_mask = diff > 10  # Threshold for detecting robot pixels

            # Apply alpha blending only where robot is present
            for c in range(3):  # RGB channels
                composite_image[:, :, c] = np.where(
                    robot_mask,
                    composite_image[:, :, c] * (1 - alpha)
                    + robot_array[:, :, c] * alpha,
                    composite_image[:, :, c],
                )

        # Convert back to uint8 and save
        final_image = np.clip(composite_image, 0, 255).astype(np.uint8)

        # Save the composite image
        iio.imwrite(save_path, final_image)
        print(f"Trajectory image saved to {save_path}")

        return final_image

    def _set_robot_visibility(self, is_visible: bool):
        """Restore original visibility states."""
        for handle in self.robot.robot_cylinder_handles:
            handle.visible = is_visible
        for handle in self.robot.robot_backbone_cylinder_handles:
            handle.visible = is_visible
