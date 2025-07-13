from typing import Sequence, Optional

import jax
import jax.numpy as jnp
import jaxls
import jaxlie
import numpy as np
import networkx as nx
from ..robots.cc_robot import CCRobot, ConstantCurvatureState
from ..geom import RobotCollision, CollGeom
from ..costs import (
    pose_cost,
    limit_cost,
    smoothness_cost,
    continuous_collision_cost,
    trajectory_length_cost,
    rest_base_cost,
    colldist_from_sdf,
)
from .ik_solver import IKSolver
from .utils import sample_states, newton_raphson
from ..geom.collision import pairwise_collide


class MotionPlanner:
    def __init__(self, robot: CCRobot, coll: RobotCollision, timesteps: int):
        self.timesteps = timesteps
        self.robot = robot
        self.coll = coll
        self.ik_solver = IKSolver(
            robot,
            num_seeds_init=10,
            num_seeds_final=1,
            total_steps=64,
            init_steps=6,
            coll=coll,
        )
        self._ik_solver_best = jax.jit(self.ik_solver.solve_ik_best_with_coll_start_end)

        self._robot_batch = jax.tree.map(lambda x: x[None], self.robot)
        self._robot_coll_batch = jax.tree.map(lambda x: x[None], self.coll)

    def start_end_interpolate(
        self,
        start_position: np.ndarray,
        start_wxyz: np.ndarray,
        end_position: np.ndarray,
        end_wxyz: np.ndarray,
        world_coll: Sequence[CollGeom],
    ):
        results: ConstantCurvatureState = self._ik_solver_best(
            start_wxyz, start_position, end_wxyz, end_position, world_coll
        )
        base_position = jnp.linspace(
            results[0].base_position, results[1].base_position, self.timesteps
        )
        theta = jnp.linspace(results[0].theta, results[1].theta, self.timesteps)
        phi = jnp.linspace(results[0].phi, results[1].phi, self.timesteps)
        return ConstantCurvatureState(base_position=base_position, theta=theta, phi=phi)

    def optimize(
        self,
        init_traj: ConstantCurvatureState,
        world_coll: Sequence[CollGeom],
    ):
        traj_vars = self.robot.var_cls(jnp.arange(self.timesteps))

        # 1. Basic regularization / limit costs.
        factors: list[jaxls.Cost] = [
            limit_cost(
                self._robot_batch,
                traj_vars,
                jnp.array([100.0])[None],
            ),
            smoothness_cost(
                self.robot.var_cls(jnp.arange(1, self.timesteps)),
                self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                jnp.array([10.0])[None],
            ),
            trajectory_length_cost(
                self._robot_batch,
                self.robot.var_cls(jnp.arange(1, self.timesteps)),
                self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                jnp.array([10.0])[None],
            ),
        ]
        # 2. Add start and end pose constraints.
        factors.extend(
            [
                jaxls.Cost(
                    lambda vals, var: ((vals[var] - init_traj[0])).flatten() * 100.0,
                    (self.robot.var_cls(jnp.arange(0, 2)),),
                    name="start_pose_constraint",
                ),
                jaxls.Cost(
                    lambda vals, var: ((vals[var] - init_traj[-1])).flatten() * 100.0,
                    (
                        self.robot.var_cls(
                            jnp.arange(self.timesteps - 2, self.timesteps)
                        ),
                    ),
                    name="end_pose_constraint",
                ),
            ]
        )
        # 3. Add collision avoidance costs.
        for world_coll_obj in world_coll:
            factors.append(
                continuous_collision_cost(
                    self._robot_batch,
                    self._robot_coll_batch,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                    self.robot.var_cls(jnp.arange(1, self.timesteps)),
                    jnp.array([20.0])[None],
                )
            )
        # 5. Solve the optimization problem.
        solution = (
            jaxls.LeastSquaresProblem(
                factors,
                [traj_vars],
            )
            .analyze()
            .solve(
                verbose=False,
                initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)),
            )
        )
        return solution[traj_vars]


class ConstrainedMotionPlanner(MotionPlanner):

    def tip_traj_follow(
        self, reference_traj: jaxlie.SE3, world_coll: Sequence[CollGeom]
    ):
        batched_ik_solver = jax.vmap(self.ik_solver.solve_ik_best)
        init_traj = batched_ik_solver(
            reference_traj.wxyz_xyz[..., :4], reference_traj.wxyz_xyz[..., 4:]
        )

        traj_vars = self.robot.var_cls(jnp.arange(self.timesteps))

        # 1. Basic regularization / limit costs.
        factors: list[jaxls.Cost] = [
            pose_cost(
                self._robot_batch,
                traj_vars,
                reference_traj,
                jnp.array([5.0])[None],
                jnp.array([1.0])[None],
            ),
            limit_cost(
                self._robot_batch,
                traj_vars,
                jnp.array([100.0])[None],
            ),
            smoothness_cost(
                self.robot.var_cls(jnp.arange(1, self.timesteps)),
                self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                jnp.array([10.0])[None],
            ),
            rest_base_cost(
                traj_vars,
                jnp.array([10.0])[None],
            ),
        ]
        # 2. Add start and end pose constraints.
        factors.extend(
            [
                jaxls.Cost(
                    lambda vals, var: ((vals[var] - init_traj[0])).flatten() * 100.0,
                    (self.robot.var_cls(jnp.arange(0, 2)),),
                    name="start_pose_constraint",
                ),
                jaxls.Cost(
                    lambda vals, var: ((vals[var] - init_traj[-1])).flatten() * 100.0,
                    (
                        self.robot.var_cls(
                            jnp.arange(self.timesteps - 2, self.timesteps)
                        ),
                    ),
                    name="end_pose_constraint",
                ),
            ]
        )
        # 3. Add collision avoidance costs.
        for world_coll_obj in world_coll:
            factors.append(
                continuous_collision_cost(
                    self._robot_batch,
                    self._robot_coll_batch,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    self.robot.var_cls(jnp.arange(0, self.timesteps - 1)),
                    self.robot.var_cls(jnp.arange(1, self.timesteps)),
                    jnp.array([50.0])[None],
                )
            )
        # 5. Solve the optimization problem.
        solution = (
            jaxls.LeastSquaresProblem(
                factors,
                [traj_vars],
            )
            .analyze()
            .solve(
                verbose=False,
                initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)),
            )
        )
        return solution[traj_vars]


class SamplingBasedMotionPlanner(ConstrainedMotionPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_root = newton_raphson(
            lambda x: x ** (self.robot.config.num_sections + 1) - x - 1, 1.0, 10_000
        )
        self.num_nearest_node = 5
        self.graph = nx.Graph()

        self.key = jax.random.PRNGKey(0)
        self.nodes_data: list[ConstantCurvatureState] = []

        self.node_similarity_threshold: float = 0.05
        self.edge_interpolation_steps: int = 10
        
        # jit
        self.distance_jit = jax.jit(self.distance)
    
    def check_collision(self,
                        cfg: ConstantCurvatureState,
                        world_coll: Sequence[CollGeom]
                    ) -> bool:
        # 1. check collision with the world
        for world_obj in world_coll:
            dist_matrix = self.coll.compute_world_collision_distance(self.robot, cfg, world_obj, 1)
            if jnp.any(dist_matrix < 0.0):
                return True

        # 2. check self collision
        # self_dist = self.coll.compute_self_collision_distance(self.robot, cfg)
        # if jnp.any(self_dist < 0.0):
        #     return True
        
        return False

    def connect_start_and_goal(self,
                               start_cfg: ConstantCurvatureState,
                               end_cfg: ConstantCurvatureState,
                               world_coll: list[CollGeom]):
        self.add_nodes_to_graph([start_cfg, end_cfg], False)

        path_segment = self.steer_and_check(start_cfg, end_cfg, world_coll)
        if path_segment is not None:
            start_node_idx = self.nodes_data.index(start_cfg)
            end_node_idx = self.nodes_data.index(end_cfg)
            cost = self.distance_jit(start_cfg, end_cfg)
            self.graph.add_edge(start_node_idx, end_node_idx, weight=cost, path_segment=path_segment)
        
            thetas = jnp.array([state.theta for state in path_segment])
            phis = jnp.array([state.phi for state in path_segment])
            base_positions = jnp.array([state.base_position for state in path_segment])
            path_segment = ConstantCurvatureState(
                base_position=base_positions,
                theta=thetas,
                phi=phis
            )
        
        return path_segment
        
    def sample_nodes_with_no_collision(self,
                                       num_samples: int,
                                       world_coll: Sequence[CollGeom]
                                    ) -> list[ConstantCurvatureState]:
        # TODO: speed up
        # sample the nodes with no collision
        collision_free_cfgs = []
        max_attempts_per_sample = 5
        num_to_sample = num_samples * max_attempts_per_sample

        sampled_batch_cfgs = sample_states(self.robot, num_to_sample, self.sample_root)

        for i in range(num_to_sample):
            if len(collision_free_cfgs) >= num_samples:
                break
            current_cfg = jax.tree_util.tree_map(lambda x: x[i], sampled_batch_cfgs)
            if not self.check_collision(current_cfg, world_coll):
                collision_free_cfgs.append(current_cfg)

        if len(collision_free_cfgs) < num_samples:
            print(
                f"Warning: Could only sample {len(collision_free_cfgs)} collision-free nodes."
            )

        return collision_free_cfgs

    def distance(
        self, cfg1: ConstantCurvatureState, cfg2: ConstantCurvatureState
    ) -> jax.Array:
        # compute the distance btw the nodes
        robot_coll_cfg1 = self.coll.at_state(self.robot, cfg1)
        robot_coll_cfg2 = self.coll.at_state(self.robot, cfg2)
        dist_matrix = pairwise_collide(robot_coll_cfg1, robot_coll_cfg2)

        return jnp.mean(dist_matrix)

    def find_k_nearest(
        self,
        k: int,
        query_cfg: ConstantCurvatureState,
        all_cfgs: list[ConstantCurvatureState],
    ) -> list[ConstantCurvatureState]:
        assert k < len(all_cfgs)
        distances = jnp.array([self.distance_jit(query_cfg, cfg) for cfg in all_cfgs])
        k_nearest_indices = jnp.argpartition(distances, k)[:k]

        return [all_cfgs[int(idx)] for idx in k_nearest_indices]

    def add_nodes_to_graph(self, cfgs: list[ConstantCurvatureState], check_duplicate = True):
        # Check for similarity to avoid adding nearly identical nodes
        # TODO： SPEED UP
        for cfg in cfgs:
            is_duplicate = False
            if check_duplicate:
                for existing_cfg in self.nodes_data:
                    if self.distance_jit(cfg, existing_cfg) < self.node_similarity_threshold:
                        is_duplicate = True
                        break
            # If the node is not a duplicate, add it to the graph
            if not is_duplicate:
                node_idx = len(self.nodes_data)
                self.nodes_data.append(cfg)
                self.graph.add_node(node_idx, cfg=cfg)

    def steer_and_check(self,
                         start_cfg: ConstantCurvatureState,
                         end_cfg: ConstantCurvatureState,
                         world_coll: Sequence[CollGeom]
                        )-> Optional[list[ConstantCurvatureState]]:
        # TODO: speed up, linspace

        interpolated_cfgs = []

        for i in range(1, self.edge_interpolation_steps + 1):
            alpha = i / self.edge_interpolation_steps

            interp_base_pos = (
                1 - alpha
            ) * start_cfg.base_position + alpha * end_cfg.base_position
            interp_theta = (1 - alpha) * start_cfg.theta + alpha * end_cfg.theta
            interp_phi = (1 - alpha) * start_cfg.phi + alpha * end_cfg.phi

            current_cfg = ConstantCurvatureState(
                base_position=interp_base_pos, theta=interp_theta, phi=interp_phi
            )

            if self.check_collision(current_cfg, world_coll):
                return None

            interpolated_cfgs.append(current_cfg)

        return interpolated_cfgs

    def build_graph(
        self,
        num_states: int,
        world_coll: list[CollGeom],
    ) -> nx.Graph:
        # Builds the PRM graph by sampling nodes and connecting them based on k-nearest neighbors.
        self.clear_graph()
        # Sample nodes with no collision
        sampled_cfgs = self.sample_nodes_with_no_collision(num_states, world_coll)
        self.add_nodes_to_graph(sampled_cfgs)

        # Add edges in the graph
        for i, current_node_cfg in enumerate(self.nodes_data):
            other_nodes = [cfg for j, cfg in enumerate(self.nodes_data) if i != j]
            k_nearest_cfgs = self.find_k_nearest(
                self.num_nearest_node, current_node_cfg, other_nodes
            )

            for nearest_cfg in k_nearest_cfgs:
                nearest_node_idx = self.nodes_data.index(nearest_cfg)
                if self.graph.has_edge(i, nearest_node_idx):
                    continue

                # check collision along the path segment
                # if not collision, use the sampled path segment as attr for path searching
                path_segment = self.steer_and_check(current_node_cfg, nearest_cfg, world_coll)
                if path_segment is not None:
                    cost = self.distance_jit(current_node_cfg, nearest_cfg)
                    self.graph.add_edge(i, nearest_node_idx, weight=cost, path_segment=path_segment)
        
        return self.graph

    def clear_graph(self):
        self.graph.clear()
        self.nodes_data.clear()

    def find_path(self,
                  start_cfg: ConstantCurvatureState,
                  end_cfg: ConstantCurvatureState,
                  sampled_nodes: int,
                  world_coll: list[CollGeom]
                ) -> Optional[list[ConstantCurvatureState]]:
        if self.check_collision(start_cfg, world_coll):
            print("Start configuration is in collision.")
            return None
        if self.check_collision(end_cfg, world_coll):
            print("End configuration is in collision.")
            return None

         # 1. build the graph if not be built
        if self.graph.number_of_nodes() == 0:
            self.build_graph(sampled_nodes, world_coll)

        # 2. try connect directly start and end
        start_end_path_segment = self.connect_start_and_goal(start_cfg, end_cfg, world_coll)
        if start_end_path_segment is not None:
            return start_end_path_segment

        # 3. find the shortest path using Dijkstra's algorithm
        start_node_idx = self.nodes_data.index(start_cfg)
        end_node_idx = self.nodes_data.index(end_cfg)
        path_node_indices = nx.shortest_path(self.graph, source=start_node_idx, target=end_node_idx, weight='weight')
        discrete_path = [self.nodes_data[idx] for idx in path_node_indices]
        
        # 4. find the shortest continuous path using interpolation
        interpolated_full_path = self.interpolate_path(discrete_path)
        return interpolated_full_path

    def interpolate_path(
        self, discrete_path: list[ConstantCurvatureState]
    ) -> Optional[list[ConstantCurvatureState]]:

        interpolated_full_path: list[ConstantCurvatureState] = [discrete_path[0]]
        for i in range(len(discrete_path) - 1):
            start_cfg_segment = discrete_path[i]
            end_cfg_segment = discrete_path[i + 1]

            start_idx = self.nodes_data.index(start_cfg_segment)
            end_idx = self.nodes_data.index(end_cfg_segment)
            edge_data = self.graph.get_edge_data(start_idx, end_idx)
            path_segment = edge_data.get("path_segment")
            interpolated_full_path.extend(path_segment)

        thetas = jnp.array([state.theta for state in interpolated_full_path])
        phis = jnp.array([state.phi for state in interpolated_full_path])
        base_positions = jnp.array(
            [state.base_position for state in interpolated_full_path]
        )
        return ConstantCurvatureState(
            base_position=base_positions, theta=thetas, phi=phis
        )
