import jax
import json
import viser
import trimesh
import numpy as np
import jax.numpy as jnp
import jaxlie
import jax_dataclasses as jdc

from ..geom.collision_cc_robot import RobotCollision
from ..geom.collision_world import WorldCollision
from ..geom.geometry import Sphere, BoundingBox
from ..robots.cc_robot import ConstantCurvatureState, CCRobot
from functools import partial


class ViserSoftRobot:
    def __init__(
        self,
        server: viser.ViserServer | viser.ClientHandle,
        robot: CCRobot,
        robot_coll: RobotCollision,
        root_node_name: str,
    ):
        self.server = server
        self.robot_coll = robot_coll
        self.root_node_name = root_node_name
        self.robot_config = robot.config
        self.cylinder_handles = []

    def _generate_section_colors(self, num_sections):
        """为每个section生成不同的颜色。"""
        colors = []
        for i in range(num_sections):
            # 使用HSV色彩空间生成均匀分布的颜色
            hue = i / num_sections
            # 转换为RGB (简化的HSV到RGB转换，S=1, V=1)
            c = 1.0
            x = c * (1 - abs((hue * 6) % 2 - 1))
            m = 0
            
            if 0 <= hue < 1/6:
                r, g, b = c, x, 0
            elif 1/6 <= hue < 2/6:
                r, g, b = x, c, 0
            elif 2/6 <= hue < 3/6:
                r, g, b = 0, c, x
            elif 3/6 <= hue < 4/6:
                r, g, b = 0, x, c
            elif 4/6 <= hue < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            colors.append((r + m, g + m, b + m))
        return colors

    def create_robot_visualizations(self):
        """创建机器人的圆柱体可视化，每个圆柱连接相邻的两个机器人节点。"""
        # 清除已有的圆柱体
        for handle in self.cylinder_handles:
            handle.remove()
        self.cylinder_handles = []
        
        # 计算机器人总节点数
        total_points = self.robot_config.num_sections * self.robot_config.num_points_per_section
        cylinder_radius = self.robot_config.radius * 0.8
        
        # 生成每个section的颜色
        section_colors = self._generate_section_colors(self.robot_config.num_sections)
        
        # 创建 n-1 个圆柱体连接 n 个节点
        for i in range(total_points - 1):
            # 确定当前圆柱体属于哪个section
            current_point_section = i // self.robot_config.num_points_per_section
            next_point_section = (i + 1) // self.robot_config.num_points_per_section
            
            # 如果两个点属于同一个section，使用该section的颜色
            # 如果跨越section边界，使用下一个section的颜色
            section_idx = max(current_point_section, next_point_section)
            section_idx = min(section_idx, self.robot_config.num_sections - 1)  # 防止越界
            
            cylinder_mesh = trimesh.creation.cylinder(
                radius=cylinder_radius,
                height=1.0,  # 单位长度，后续通过变换调整
                sections=16
            )
            
            # 设置圆柱体颜色
            color = section_colors[section_idx]
            cylinder_mesh.visual.face_colors = [int(c * 255) for c in color] + [255]  # RGBA
            
            cylinder_handle = self.server.scene.add_mesh_trimesh(
                name=f"{self.root_node_name}/cylinder_{i}",
                mesh=cylinder_mesh,
            )
            self.cylinder_handles.append(cylinder_handle)

    def update_pose(self, all_poses):
        """根据机器人状态更新圆柱体的位置和方向。
        
        Args:
            all_poses: 机器人所有节点的SE3位姿矩阵，形状为 (n_points, 4, 4)
        """
        if not self.cylinder_handles:
            return
        
        for i, handle in enumerate(self.cylinder_handles):
            # 获取相邻两个节点的位置
            pos1 = jaxlie.SE3.from_matrix(all_poses[i]).translation()
            pos2 = jaxlie.SE3.from_matrix(all_poses[i + 1]).translation()
            
            # 计算圆柱体的中心位置和方向
            center = (pos1 + pos2) / 2.0
            direction = pos2 - pos1
            length = np.linalg.norm(direction)
            
            if length > 1e-6:
                # 计算从Z轴到目标方向的旋转
                z_axis = np.array([0., 0., 1.])
                direction_unit = direction / length
                rotation = self._compute_rotation_z_to_direction(z_axis, direction_unit)
                
                # 检查旋转是否有效
                if np.any(np.isnan(rotation.wxyz)):
                    print(f"Warning: NaN rotation for cylinder {i}")
                    rotation = jaxlie.SO3.identity()
                
                # 更新圆柱体的位置、方向和长度
                handle.position = tuple(center)
                handle.wxyz = tuple(rotation.wxyz)
                handle.scale = (1.0, 1.0, float(length) * 0.6)
            else:
                # 长度太小，隐藏圆柱体
                handle.scale = (0.0, 0.0, 0.0)
    
    def _compute_rotation_z_to_direction(self, z_axis, target_direction):
        """计算从Z轴到目标方向的旋转四元数。"""
        # 检查是否平行
        dot_product = np.dot(z_axis, target_direction)
        if np.abs(dot_product) > 0.999:
            # 平行或反平行
            if dot_product > 0:
                return jaxlie.SO3.identity()
            else:
                # 180度旋转，选择X轴作为旋转轴
                return jaxlie.SO3.from_matrix(np.diag([1, -1, -1]))
        
        # 计算旋转轴和角度
        rotation_axis = np.cross(z_axis, target_direction)
        axis_norm = np.linalg.norm(rotation_axis)
        
        # 防止除零错误
        if axis_norm < 1e-8:
            return jaxlie.SO3.identity()
            
        rotation_axis = rotation_axis / axis_norm
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # 使用轴角表示创建旋转
        return jaxlie.SO3.from_quaternion_xyzw(
            np.concatenate([rotation_axis * np.sin(angle/2), [np.cos(angle/2)]])
        )

    def create_collision_visualizations(self):
        """Create sphere visualizations for the robot collision model."""
        if not hasattr(self, 'sphere_handles'):
            self.sphere_handles = []
            
        for handle in self.sphere_handles:
            handle.remove()
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
                        jnp.array([0, 0, 0]),
                        jnp.array([self.robot_coll.coll.radius[i]]),
                    ).to_trimesh(),
                )
                self.sphere_handles.append(sphere_handle)

    def update_collision_pose(self, all_poses: jnp.ndarray):
        """Update collision sphere positions based on robot state."""
        if not hasattr(self, 'sphere_handles'):
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
                    "center": [round(float(center), 2) for center in centers[i]],
                    "radius": round(float(radii[i]), 2),
                }
        elif isinstance(self.world_coll.obstacles, BoundingBox):
            obstacles_dict = {}
            centers = self.world_coll.obstacles.pose.translation()
            extents = self.world_coll.obstacles.extents
            for i in range(len(centers)):
                obstacles_dict[f"obstacle_{i+1}"] = {
                    "type": "bbox",
                    "center": [round(float(center), 2) for center in centers[i]],
                    "extents": [round(float(extent), 2) for extent in extents[i]],
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
            self.server.scene.add_mesh_trimesh(
                name=f"obstacles/collision",
                mesh=self.world_coll.obstacles.to_trimesh(),
            )

        # add ground visualizations
        self.server.scene.add_grid("/ground", width=6, height=6)
