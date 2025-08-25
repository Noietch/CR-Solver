"""
Optimized Probabilistic Roadmap (PRM) Motion Planner for CC Robot
Enhanced JAX-based parallel implementation for maximum performance
"""

from typing import Optional, Sequence, Dict, List, Tuple
import jax
import jax.tree_util
import jax.numpy as jnp
import networkx as nx
from dataclasses import dataclass

from ...robots.cc_robot import CCRobot, ConstantCurvatureState
from ...geom import RobotCollision, CollGeom
from .utils import HPolyhedronSampler


@dataclass
class PRMOptions:
    """Options for PRM planner"""

    max_neighbors: int = 10
    max_edge_distance: float = 2.0
    edge_interpolation_steps: int = 2
    max_num_blocked_edges_before_discard: int = 3
    max_planning_attempts: int = 3
    batch_size: int = 1000
    parallel_edge_checks: int = 100  # Number of edges to check in parallel


class ParallelPRM:
    """
    Highly optimized PRM planner using advanced JAX features.
    Key optimizations:
    - Fully vectorized edge collision checking
    - Parallel k-NN search with JAX ops
    - Optimized trajectory interpolation with scan
    - Batch matrix operations for distance computation
    """

    def __init__(
        self,
        robot: CCRobot,
        robot_coll: RobotCollision,
        options: Optional[PRMOptions] = None,
    ):
        self.robot = robot
        self.robot_coll = robot_coll
        self.options = options or PRMOptions()

        # Graph structure
        self.graph = nx.Graph()
        self.nodes: List[ConstantCurvatureState] = []
        self.node_to_idx: Dict[int, int] = {}

        # Collision tracking
        self.collision_set: set = set()
        self.forbidden_edges: set = set()
        self.edge_attempt_count: Dict[int, int] = {}

        # Initialize HPolyhedron sampler
        self.sampler = HPolyhedronSampler(robot, seed=42)

        # Enhanced JIT compiled functions with optimizations
        self._batch_distance = jax.jit(self._batch_distance_fn)
        self._batch_interpolate = jax.jit(
            self._batch_interpolate_fn, static_argnums=(2,)
        )
        self._batch_check_collision = jax.jit(self._batch_check_collision_fn)
        self._compute_all_distances = jax.jit(self._compute_all_distances_fn)
        self._parallel_check_edges = jax.jit(
            self._parallel_check_edges_fn, static_argnums=(2,)
        )
        self._find_k_nearest = jax.jit(self._find_k_nearest_fn, static_argnums=(1,))
        self._scan_interpolate = jax.jit(self._scan_interpolate_fn)

        # Random key
        self.key = jax.random.PRNGKey(42)

    def clear(self):
        """Clear the roadmap"""
        self.graph.clear()
        self.nodes.clear()
        self.node_to_idx.clear()
        self.collision_set.clear()
        self.forbidden_edges.clear()
        self.edge_attempt_count.clear()

    def add_node(self, state: ConstantCurvatureState) -> int:
        """Add a node to the roadmap"""
        idx = len(self.nodes)
        self.nodes.append(state)
        self.node_to_idx[id(state)] = idx
        self.graph.add_node(idx, state=state)
        return idx

    def sample_collision_free_nodes(
        self, num_samples: int, world_coll: Sequence[CollGeom]
    ) -> List[ConstantCurvatureState]:
        """
        Optimized batch sampling with parallel collision checking.
        """
        collision_free = []
        batch_size = min(self.options.batch_size * 2, num_samples * 2)  # Larger batches
        mixing_steps = 3  # Reduced for speed

        # Start from Chebyshev center
        current_states = [self.sampler.get_feasible_point() for _ in range(batch_size)]

        num_attempts = 0
        max_attempts = num_samples * 3  # More efficient, so fewer attempts needed

        while len(collision_free) < num_samples and num_attempts < max_attempts:
            # Generate batch of samples in parallel
            batch_states = []
            for i in range(min(batch_size, (num_samples - len(collision_free)) * 2)):
                new_state = self.sampler.uniform_sample(
                    current_states[i % len(current_states)], mixing_steps
                )
                batch_states.append(new_state)
                current_states[i % len(current_states)] = new_state

            num_attempts += len(batch_states)

            if batch_states:
                # Vectorized collision check
                batched = jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(xs), *batch_states
                )
                collision_mask = self._batch_check_collision(batched, world_coll)

                # Extract collision-free states efficiently
                valid_indices = jnp.where(~collision_mask)[0]
                for idx in valid_indices:
                    collision_free.append(batch_states[int(idx)])
                    if len(collision_free) >= num_samples:
                        break

        return collision_free[:num_samples]

    def _batch_check_collision_fn(
        self, states: ConstantCurvatureState, world_coll: Sequence[CollGeom]
    ) -> jnp.ndarray:
        """Optimized batch collision checking"""

        # Vectorized collision check for all world objects at once
        def check_single_object(world_obj):
            dist_matrix = jax.vmap(
                lambda s: self.robot_coll.compute_world_collision_distance(
                    self.robot, s, world_obj, 1
                )
            )(states)
            return jnp.any(dist_matrix < 0.0, axis=(1, 2))

        # Stack collision results
        collision_masks = jnp.stack([check_single_object(obj) for obj in world_coll])
        return jnp.any(collision_masks, axis=0)

    def _batch_distance_fn(
        self, state1: ConstantCurvatureState, states2: ConstantCurvatureState
    ) -> jnp.ndarray:
        """Vectorized distance computation"""
        # Use squared distances to avoid sqrt when not needed
        theta_dist_sq = jnp.sum((states2.theta - state1.theta[None, :]) ** 2, axis=1)
        phi_dist_sq = jnp.sum((states2.phi - state1.phi[None, :]) ** 2, axis=1)

        return jnp.sqrt(theta_dist_sq + phi_dist_sq)

    def _batch_interpolate_fn(
        self,
        state1: ConstantCurvatureState,
        state2: ConstantCurvatureState,
        num_steps: int,
    ) -> ConstantCurvatureState:
        """Optimized interpolation using vectorized operations"""
        alphas = jnp.linspace(0, 1, num_steps + 2)[1:-1]

        # Vectorized interpolation
        base_positions = jnp.zeros((len(alphas), 3))
        thetas = jnp.outer(1 - alphas, state1.theta) + jnp.outer(alphas, state2.theta)
        phis = jnp.outer(1 - alphas, state1.phi) + jnp.outer(alphas, state2.phi)

        return ConstantCurvatureState(
            base_position=base_positions, theta=thetas, phi=phis
        )

    def _compute_all_distances_fn(
        self, all_states: ConstantCurvatureState
    ) -> jnp.ndarray:
        """Optimized pairwise distance computation using broadcasting"""
        # Efficient squared distance computation
        theta_diff = all_states.theta[:, None, :] - all_states.theta[None, :, :]
        phi_diff = all_states.phi[:, None, :] - all_states.phi[None, :, :]

        # Combined squared distances
        dist_sq = jnp.sum(theta_diff**2, axis=2) + jnp.sum(phi_diff**2, axis=2)
        return jnp.sqrt(dist_sq)

    def _find_k_nearest_fn(self, distances: jnp.ndarray, k: int) -> jnp.ndarray:
        """Efficiently find k-nearest neighbors for all nodes"""
        n = distances.shape[0]

        # Set diagonal to inf to exclude self
        distances_masked = distances.at[jnp.diag_indices(n)].set(jnp.inf)

        # Use top-k for better performance
        if k < n - 1:
            # Get k smallest indices for each row
            _, indices = jax.lax.top_k(-distances_masked, k)
        else:
            # All neighbors except self
            indices = jnp.argsort(distances_masked, axis=1)[:, :k]

        return indices

    def _parallel_check_edges_fn(
        self,
        state_pairs: Tuple[ConstantCurvatureState, ConstantCurvatureState],
        world_coll: Sequence[CollGeom],
        steps: int,
    ) -> jnp.ndarray:
        """Check multiple edges in parallel"""
        states1, states2 = state_pairs
        num_edges = states1.theta.shape[0]

        # Vectorized interpolation for all edges
        def interpolate_edge(s1, s2):
            alphas = jnp.linspace(0, 1, steps + 2)[1:-1]
            thetas = jnp.outer(1 - alphas, s1.theta) + jnp.outer(alphas, s2.theta)
            phis = jnp.outer(1 - alphas, s1.phi) + jnp.outer(alphas, s2.phi)
            return thetas, phis

        # Vectorized over all edges
        all_thetas, all_phis = jax.vmap(interpolate_edge)(states1, states2)

        # Reshape for batch collision checking
        all_thetas_flat = all_thetas.reshape(-1, all_thetas.shape[-1])
        all_phis_flat = all_phis.reshape(-1, all_phis.shape[-1])
        base_positions_flat = jnp.zeros((all_thetas_flat.shape[0], 3))

        intermediate = ConstantCurvatureState(
            base_position=base_positions_flat, theta=all_thetas_flat, phi=all_phis_flat
        )

        # Batch collision check
        collision_mask = self._batch_check_collision(intermediate, world_coll)

        # Reshape and check if any intermediate point has collision
        collision_mask_reshaped = collision_mask.reshape(num_edges, steps)
        edge_has_collision = jnp.any(collision_mask_reshaped, axis=1)

        return ~edge_has_collision  # Return True for valid edges

    def _scan_interpolate_fn(
        self, waypoints: List[ConstantCurvatureState]
    ) -> ConstantCurvatureState:
        """Use JAX scan for efficient trajectory building"""

        def interpolate_segment(_, waypoint_pair):
            prev_state, curr_state = waypoint_pair

            # Interpolate between waypoints
            alphas = jnp.linspace(0, 1, self.options.edge_interpolation_steps + 2)
            thetas = jnp.outer(1 - alphas, prev_state.theta) + jnp.outer(
                alphas, curr_state.theta
            )
            phis = jnp.outer(1 - alphas, prev_state.phi) + jnp.outer(
                alphas, curr_state.phi
            )
            base_positions = jnp.zeros((len(alphas), 3))

            segment = ConstantCurvatureState(
                base_position=base_positions, theta=thetas, phi=phis
            )
            return curr_state, segment

        # Create pairs of consecutive waypoints
        waypoint_pairs = [
            (waypoints[i], waypoints[i + 1]) for i in range(len(waypoints) - 1)
        ]

        # Use scan for sequential processing
        _, segments = jax.lax.scan(
            interpolate_segment, waypoints[0], jnp.array(waypoint_pairs)
        )

        # Concatenate all segments
        return jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), *segments
        )

    def build_roadmap(self, num_nodes: int, world_coll: Sequence[CollGeom]):
        """Build optimized PRM roadmap"""
        print(f"Building optimized roadmap with {num_nodes} nodes...")

        # Sample collision-free nodes
        nodes = self.sample_collision_free_nodes(num_nodes, world_coll)

        # Add nodes to graph
        node_indices = []
        for node in nodes:
            idx = self.add_node(node)
            node_indices.append(idx)

        print(f"Added {len(nodes)} collision-free nodes")

        # Build edges using optimized k-nearest neighbors
        self._connect_nearest_neighbors(node_indices, world_coll)

        print(f"Roadmap built with {self.graph.number_of_edges()} edges")

    def _connect_nearest_neighbors(
        self, node_indices: List[int], world_coll: Sequence[CollGeom]
    ):
        """Optimized edge connection with parallel processing"""
        num_nodes = len(node_indices)
        if num_nodes < 2:
            return

        # Stack all nodes
        all_states = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *[self.nodes[idx] for idx in node_indices]
        )

        # Compute all distances at once
        print(f"Computing pairwise distances for {num_nodes} nodes...")
        all_distances = self._compute_all_distances(all_states)

        # Find k-nearest neighbors for all nodes in parallel
        k = min(self.options.max_neighbors, num_nodes - 1)
        nearest_indices = self._find_k_nearest(all_distances, k)

        # Collect potential edges
        edges_to_check = []
        edge_states1 = []
        edge_states2 = []

        for i, idx1 in enumerate(node_indices):
            for j_local in nearest_indices[i]:
                j = int(j_local)
                if i >= j:  # Avoid duplicates
                    continue

                idx2 = node_indices[j]
                dist = float(all_distances[i, j])

                if dist > self.options.max_edge_distance:
                    continue
                if self.graph.has_edge(idx1, idx2):
                    continue
                if (idx1, idx2) in self.forbidden_edges:
                    continue

                edges_to_check.append((idx1, idx2, dist))
                edge_states1.append(self.nodes[idx1])
                edge_states2.append(self.nodes[idx2])

        # Parallel edge collision checking
        print(f"Checking {len(edges_to_check)} potential edges in parallel...")

        if edges_to_check:
            # Process in batches for memory efficiency
            batch_size = self.options.parallel_edge_checks
            valid_edges_mask = []

            for batch_start in range(0, len(edges_to_check), batch_size):
                batch_end = min(batch_start + batch_size, len(edges_to_check))

                # Stack batch states
                batch_states1 = jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(xs), *edge_states1[batch_start:batch_end]
                )
                batch_states2 = jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(xs), *edge_states2[batch_start:batch_end]
                )

                # Parallel collision check
                batch_valid = self._parallel_check_edges(
                    (batch_states1, batch_states2),
                    world_coll,
                    self.options.edge_interpolation_steps,
                )

                valid_edges_mask.extend(batch_valid)

            # Add valid edges to graph
            for (idx1, idx2, dist), is_valid in zip(edges_to_check, valid_edges_mask):
                if is_valid:
                    self.graph.add_edge(idx1, idx2, weight=dist)

    def find_path(
        self,
        start: ConstantCurvatureState,
        goal: ConstantCurvatureState,
        world_coll: Sequence[CollGeom],
    ) -> Optional[ConstantCurvatureState]:
        """
        Find path with optimized collision checking and interpolation.
        """
        # Check start and goal for collision
        start_coll = self._batch_check_collision(
            jax.tree_util.tree_map(lambda x: x[None, ...], start), world_coll
        )[0]
        goal_coll = self._batch_check_collision(
            jax.tree_util.tree_map(lambda x: x[None, ...], goal), world_coll
        )[0]

        if start_coll:
            print("Start configuration is in collision")
            return None
        if goal_coll:
            print("Goal configuration is in collision")
            return None

        # Multiple planning attempts
        for attempt in range(self.options.max_planning_attempts):
            result = self._plan_attempt(start, goal, world_coll)
            if result is not None:
                print(f"Path found on attempt {attempt + 1}")
                return result

        print("Failed to find path after all attempts")
        return None

    def _plan_attempt(
        self,
        start: ConstantCurvatureState,
        goal: ConstantCurvatureState,
        world_coll: Sequence[CollGeom],
    ) -> Optional[ConstantCurvatureState]:
        """Optimized planning attempt"""

        # Find closest nodes efficiently
        start_idx = self._find_closest_valid_node(start)
        goal_idx = self._find_closest_valid_node(goal)

        if start_idx is None or goal_idx is None:
            return None

        # Verify connections
        start_node = self.nodes[start_idx]
        goal_node = self.nodes[goal_idx]

        # Check edge validity
        start_edge_valid = self._parallel_check_edges(
            (
                jax.tree_util.tree_map(lambda x: x[None, ...], start),
                jax.tree_util.tree_map(lambda x: x[None, ...], start_node),
            ),
            world_coll,
            self.options.edge_interpolation_steps,
        )[0]

        if not start_edge_valid:
            self._mark_node_collision(start_idx)
            return None

        goal_edge_valid = self._parallel_check_edges(
            (
                jax.tree_util.tree_map(lambda x: x[None, ...], goal_node),
                jax.tree_util.tree_map(lambda x: x[None, ...], goal),
            ),
            world_coll,
            self.options.edge_interpolation_steps,
        )[0]

        if not goal_edge_valid:
            self._mark_node_collision(goal_idx)
            return None

        # Find path in roadmap
        try:
            path_indices = nx.shortest_path(
                self.graph, start_idx, goal_idx, weight="weight"
            )
        except nx.NetworkXNoPath:
            return None

        # Validate path with parallel edge checking
        if not self._validate_path(path_indices, world_coll):
            return None

        # Build trajectory efficiently
        return self._build_trajectory(start, goal, path_indices)

    def _find_closest_valid_node(self, state: ConstantCurvatureState) -> Optional[int]:
        """Find closest node using vectorized operations"""
        if not self.nodes:
            return None

        valid_indices = [
            i for i in range(len(self.nodes)) if i not in self.collision_set
        ]

        if not valid_indices:
            return None

        # Stack valid states
        valid_states = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *[self.nodes[i] for i in valid_indices]
        )

        # Compute distances in parallel
        distances = self._batch_distance(state, valid_states)
        closest_idx = valid_indices[int(jnp.argmin(distances))]

        return closest_idx

    def _validate_path(
        self, path_indices: List[int], world_coll: Sequence[CollGeom]
    ) -> bool:
        """Validate path with batch edge checking"""
        if len(path_indices) < 2:
            return True

        # Collect all edges to check
        edges_to_validate = []
        edge_states1 = []
        edge_states2 = []

        for i in range(len(path_indices) - 1):
            idx1, idx2 = path_indices[i], path_indices[i + 1]

            if (idx1, idx2) in self.forbidden_edges:
                return False

            edges_to_validate.append((idx1, idx2))
            edge_states1.append(self.nodes[idx1])
            edge_states2.append(self.nodes[idx2])

        if not edges_to_validate:
            return True

        # Batch collision check
        states1 = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *edge_states1)
        states2 = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *edge_states2)

        valid_mask = self._parallel_check_edges(
            (states1, states2), world_coll, self.options.edge_interpolation_steps
        )

        # Update forbidden edges for invalid ones
        for (idx1, idx2), is_valid in zip(edges_to_validate, valid_mask):
            if not is_valid:
                self.forbidden_edges.add((idx1, idx2))
                self.forbidden_edges.add((idx2, idx1))
                self._update_edge_attempts(idx1)
                return False

        return True

    def _build_trajectory(
        self,
        start: ConstantCurvatureState,
        goal: ConstantCurvatureState,
        path_indices: List[int],
    ) -> ConstantCurvatureState:
        """Build trajectory using optimized interpolation"""
        # Collect all waypoints
        waypoints = [start]
        waypoints.extend([self.nodes[idx] for idx in path_indices])
        waypoints.append(goal)

        # Use vectorized interpolation for all segments
        trajectory_segments = []

        for i in range(len(waypoints) - 1):
            # Add current waypoint
            trajectory_segments.append(
                jax.tree_util.tree_map(lambda x: x[None, ...], waypoints[i])
            )

            # Interpolate to next waypoint
            segment = self._batch_interpolate(
                waypoints[i], waypoints[i + 1], self.options.edge_interpolation_steps
            )
            trajectory_segments.append(segment)

        # Add final waypoint
        trajectory_segments.append(
            jax.tree_util.tree_map(lambda x: x[None, ...], waypoints[-1])
        )

        # Concatenate all segments efficiently
        full_trajectory = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), *trajectory_segments
        )

        return full_trajectory

    def _mark_node_collision(self, node_idx: int):
        """Mark a node as being in collision"""
        self.collision_set.add(node_idx)
        neighbors = list(self.graph.neighbors(node_idx))
        for neighbor in neighbors:
            self.graph.remove_edge(node_idx, neighbor)

    def _update_edge_attempts(self, node_idx: int):
        """Update edge attempt count"""
        self.edge_attempt_count[node_idx] = self.edge_attempt_count.get(node_idx, 0) + 1

        if (
            self.edge_attempt_count[node_idx]
            >= self.options.max_num_blocked_edges_before_discard
        ):
            self._mark_node_collision(node_idx)

    def save_roadmap(self, filepath: str):
        """Save roadmap to file"""
        import pickle

        roadmap_data = {
            "nodes": self.nodes,
            "edges": list(self.graph.edges(data=True)),
            "collision_set": self.collision_set,
            "forbidden_edges": self.forbidden_edges,
        }
        with open(filepath, "wb") as f:
            pickle.dump(roadmap_data, f)

    def load_roadmap(self, filepath: str):
        """Load roadmap from file"""
        import pickle

        with open(filepath, "rb") as f:
            roadmap_data = pickle.load(f)

        self.clear()

        for node in roadmap_data["nodes"]:
            self.add_node(node)

        for u, v, data in roadmap_data["edges"]:
            self.graph.add_edge(u, v, **data)

        self.collision_set = roadmap_data["collision_set"]
        self.forbidden_edges = roadmap_data["forbidden_edges"]
