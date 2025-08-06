"""
Probabilistic Roadmap (PRM) Motion Planner for CC Robot
JAX-based parallel implementation for efficient roadmap construction
"""

from typing import Optional, Sequence, Dict, List
import jax
import jax.numpy as jnp
import networkx as nx
from dataclasses import dataclass

from ...robots.cc_robot import CCRobot, ConstantCurvatureState
from ...geom import RobotCollision, CollGeom
from .hpolyhedron_sampler import HPolyhedronSampler


@dataclass
class PRMOptions:
    """Options for PRM planner"""

    max_neighbors: int = 10
    max_edge_distance: float = 2.0
    edge_interpolation_steps: int = 5  # Reduced for speed
    node_similarity_threshold: float = 0.1
    max_iterations_steering: int = 10
    max_nodes_to_expand: int = 10000
    voxel_padding: float = 0.02
    online_edge_step_size: float = 0.1
    max_num_blocked_edges_before_discard: int = 3
    max_planning_attempts: int = 5


class ParallelPRM:
    """
    Efficient PRM planner using JAX for parallel computation.
    Builds roadmaps with lazy collision checking and edge invalidation.
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

        # Pre-compile collision checking for efficiency
        self._compute_single_collision = jax.jit(
            lambda robot, robot_coll, state, world_obj: robot_coll.compute_world_collision_distance(
                robot, state, world_obj, 1
            )
        )

        # JIT compiled functions
        self._batch_distance = jax.jit(self._batch_distance_fn)
        self._batch_interpolate = jax.jit(
            self._batch_interpolate_fn, static_argnums=(2,)
        )

        # Cache for compiled collision functions
        self._compiled_funcs = {}

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
        Sample collision-free configurations using batch sampling for speed.
        """
        collision_free = []
        batch_size = min(100, num_samples)  # Larger batches for efficiency
        mixing_steps = 5  # Fewer mixing steps for speed

        # Start from Chebyshev center
        current_state = self.sampler.get_feasible_point()

        num_attempts = 0
        max_attempts = num_samples * 5  # Reduced multiplier since we're more efficient

        while len(collision_free) < num_samples and num_attempts < max_attempts:
            # Generate a batch of samples
            batch_states = []
            for _ in range(min(batch_size, num_samples - len(collision_free))):
                new_state = self.sampler.uniform_sample(current_state, mixing_steps)
                batch_states.append(new_state)
                current_state = new_state
                num_attempts += 1

            # Batch collision check
            if batch_states:
                # Stack states for batch processing
                batched = jax.tree.map(lambda *xs: jnp.stack(xs), *batch_states)
                collision_mask = self._batch_check_collision(batched, world_coll)

                # Extract collision-free states
                for i, is_collision in enumerate(collision_mask):
                    if not is_collision:
                        collision_free.append(batch_states[i])
                        if len(collision_free) >= num_samples:
                            break

        if len(collision_free) < num_samples:
            print(
                f"Warning: Only sampled {len(collision_free)}/{num_samples} collision-free nodes"
            )

        return collision_free[:num_samples]

    def _check_single_collision(
        self, state: ConstantCurvatureState, world_coll: Sequence[CollGeom]
    ) -> bool:
        """Check collision for a single state"""
        state_batch = state.repeat(1, axis=0)
        collision_mask = self._batch_check_collision(state_batch, world_coll)
        return bool(collision_mask[0])

    def _batch_check_collision(
        self, states: ConstantCurvatureState, world_coll: Sequence[CollGeom]
    ) -> jnp.ndarray:
        """Batch collision checking for multiple states"""
        num_states = states.base_position.shape[0]
        collision_mask = jnp.zeros(num_states, dtype=bool)

        # Check collision with each world object
        for world_obj in world_coll:
            dist_matrix = jax.vmap(
                lambda s: self.robot_coll.compute_world_collision_distance(
                    self.robot, s, world_obj, 1
                )
            )(states)

            # Mark states with collision
            has_collision = jnp.any(dist_matrix < 0.0, axis=(1, 2))
            collision_mask = collision_mask | has_collision

        return collision_mask

    def _batch_distance_fn(
        self, state1: ConstantCurvatureState, states2: ConstantCurvatureState
    ) -> jnp.ndarray:
        """Compute distances from one state to many states.
        Only considers arm parameters since base is fixed.
        """
        # Only arm configuration distance (no base)
        theta_dist = jnp.linalg.norm(states2.theta - state1.theta[None, :], axis=1)
        phi_dist = jnp.linalg.norm(states2.phi - state1.phi[None, :], axis=1)

        # Weighted combination of arm parameters only
        return theta_dist + phi_dist

    def _batch_interpolate_fn(
        self,
        state1: ConstantCurvatureState,
        state2: ConstantCurvatureState,
        num_steps: int,
    ) -> ConstantCurvatureState:
        """Interpolate between two states.
        Base position remains fixed at origin.
        """
        alphas = jnp.linspace(0, 1, num_steps + 2)[1:-1]  # Exclude endpoints

        # Fixed base at origin
        base_positions = jnp.zeros((len(alphas), 3))

        # Only interpolate arm parameters
        thetas = (
            state1.theta[None, :] * (1 - alphas[:, None])
            + state2.theta[None, :] * alphas[:, None]
        )
        phis = (
            state1.phi[None, :] * (1 - alphas[:, None])
            + state2.phi[None, :] * alphas[:, None]
        )

        return ConstantCurvatureState(
            base_position=base_positions, theta=thetas, phi=phis
        )

    def _compute_all_distances(self, all_states: ConstantCurvatureState) -> jnp.ndarray:
        """Compute all pairwise distances between states"""
        n = all_states.theta.shape[0]

        # Compute pairwise distances for theta and phi
        theta_diff = all_states.theta[:, None, :] - all_states.theta[None, :, :]
        phi_diff = all_states.phi[:, None, :] - all_states.phi[None, :, :]

        theta_dist = jnp.linalg.norm(theta_diff, axis=2)
        phi_dist = jnp.linalg.norm(phi_diff, axis=2)

        return theta_dist + phi_dist

    def _batch_check_edges(
        self,
        edges_to_check: List,
        all_states: ConstantCurvatureState,
        world_coll: Sequence[CollGeom],
    ) -> List[bool]:
        """Batch check multiple edges for collision"""
        if not edges_to_check:
            return []

        valid_edges = []
        batch_size = 10  # Process edges in batches

        for batch_start in range(0, len(edges_to_check), batch_size):
            batch_end = min(batch_start + batch_size, len(edges_to_check))
            batch = edges_to_check[batch_start:batch_end]

            for idx1, idx2, dist, i, j in batch:
                # Get states
                state1 = jax.tree.map(lambda x: x[i], all_states)
                state2 = jax.tree.map(lambda x: x[j], all_states)

                # Interpolate edge
                intermediate = self._batch_interpolate(
                    state1, state2, self.options.edge_interpolation_steps
                )

                # Check collision
                collision_mask = self._batch_check_collision(intermediate, world_coll)
                has_collision = jnp.any(collision_mask)

                valid_edges.append(not has_collision)

        return valid_edges

    def _check_edge_collision(
        self,
        state1: ConstantCurvatureState,
        state2: ConstantCurvatureState,
        world_coll: Sequence[CollGeom],
    ) -> bool:
        """Check collision along an edge using interpolation"""
        # Interpolate intermediate states
        intermediate = self._batch_interpolate(
            state1, state2, self.options.edge_interpolation_steps
        )

        # Check collision for all intermediate states
        collision_mask = self._batch_check_collision(intermediate, world_coll)

        return jnp.any(collision_mask)

    def build_roadmap(self, num_nodes: int, world_coll: Sequence[CollGeom]):
        """Build the PRM roadmap"""
        print(f"Building roadmap with {num_nodes} nodes...")

        # Sample collision-free nodes
        nodes = self.sample_collision_free_nodes(num_nodes, world_coll)

        # Add nodes to graph
        node_indices = []
        for node in nodes:
            idx = self.add_node(node)
            node_indices.append(idx)

        print(f"Added {len(nodes)} collision-free nodes")

        # Build edges using k-nearest neighbors
        self._connect_nearest_neighbors(node_indices, world_coll)

        print(f"Roadmap built with {self.graph.number_of_edges()} edges")

    def _connect_nearest_neighbors(
        self, node_indices: List[int], world_coll: Sequence[CollGeom]
    ):
        """Connect nodes to their k-nearest neighbors using batch processing"""
        num_nodes = len(node_indices)
        if num_nodes < 2:
            return

        # Stack all nodes for batch processing
        all_states = jax.tree.map(
            lambda *xs: jnp.stack(xs), *[self.nodes[idx] for idx in node_indices]
        )

        # Compute all pairwise distances at once
        print(f"Computing pairwise distances for {num_nodes} nodes...")
        all_distances = self._compute_all_distances(all_states)

        # Process connections
        edges_to_check = []
        k = min(self.options.max_neighbors, num_nodes - 1)

        for i, idx1 in enumerate(node_indices):
            if i % 100 == 0 and i > 0:
                print(f"Processing node {i}/{num_nodes}")

            # Get k-nearest neighbors for this node
            distances_i = all_distances[i]
            # Mask out self-distance
            distances_i = distances_i.at[i].set(jnp.inf)

            # Find k nearest
            if k < len(distances_i):
                nearest_indices = jnp.argpartition(distances_i, k)[:k]
            else:
                nearest_indices = jnp.arange(len(distances_i))
                nearest_indices = nearest_indices[nearest_indices != i]

            for j in nearest_indices:
                idx2 = node_indices[j]
                dist = float(distances_i[j])

                # Skip if too far or edge exists
                if dist > self.options.max_edge_distance:
                    continue
                if idx1 >= idx2:  # Avoid duplicate checks
                    continue
                if self.graph.has_edge(idx1, idx2):
                    continue
                if (idx1, idx2) in self.forbidden_edges:
                    continue

                edges_to_check.append((idx1, idx2, dist, i, j))

        # Batch check edge collisions
        print(f"Checking {len(edges_to_check)} potential edges...")
        valid_edges = self._batch_check_edges(edges_to_check, all_states, world_coll)

        # Add valid edges to graph
        for (idx1, idx2, dist, i, j), is_valid in zip(
            edges_to_check[: len(valid_edges)], valid_edges
        ):
            if is_valid:
                self.graph.add_edge(idx1, idx2, weight=dist)

    def find_path(
        self,
        start: ConstantCurvatureState,
        goal: ConstantCurvatureState,
        world_coll: Sequence[CollGeom],
    ) -> Optional[ConstantCurvatureState]:
        """
        Find a path from start to goal configuration.
        Uses lazy collision checking and replanning strategy.
        """
        # Check start and goal for collision
        start_coll = self._batch_check_collision(start.repeat(1, axis=0), world_coll)[0]
        goal_coll = self._batch_check_collision(goal.repeat(1, axis=0), world_coll)[0]

        if start_coll:
            print("Start configuration is in collision")
            return None
        if goal_coll:
            print("Goal configuration is in collision")
            return None

        # Multiple planning attempts with edge invalidation
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
        """Single planning attempt with lazy collision checking"""

        # Find closest nodes to start and goal
        start_idx = self._find_closest_valid_node(start)
        goal_idx = self._find_closest_valid_node(goal)

        if start_idx is None or goal_idx is None:
            return None

        # Check if we can steer to/from roadmap
        # start_node = self.nodes[start_idx]
        # goal_node = self.nodes[goal_idx]

        # if self._check_edge_collision(start, start_node, world_coll):
        #     self._mark_node_collision(start_idx)
        #     return None

        # if self._check_edge_collision(goal_node, goal, world_coll):
        #     self._mark_node_collision(goal_idx)
        #     return None

        # Find path in roadmap
        try:
            path_indices = nx.shortest_path(
                self.graph, start_idx, goal_idx, weight="weight"
            )
        except nx.NetworkXNoPath:
            return None

        # Lazy collision checking along path
        valid_path = self._validate_path(path_indices, world_coll)
        if not valid_path:
            return None

        # Build full trajectory
        return self._build_trajectory(start, goal, path_indices)

    def _find_closest_valid_node(self, state: ConstantCurvatureState) -> Optional[int]:
        """Find closest non-colliding node in roadmap"""
        if not self.nodes:
            return None

        # Stack all valid nodes
        valid_indices = [
            i for i in range(len(self.nodes)) if i not in self.collision_set
        ]

        if not valid_indices:
            return None

        valid_states = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *[self.nodes[i] for i in valid_indices]
        )

        # Find closest
        distances = self._batch_distance(state, valid_states)
        closest_idx = valid_indices[int(jnp.argmin(distances))]

        return closest_idx

    def _validate_path(
        self, path_indices: List[int], world_coll: Sequence[CollGeom]
    ) -> bool:
        """Validate path edges with lazy collision checking"""
        for i in range(len(path_indices) - 1):
            idx1, idx2 = path_indices[i], path_indices[i + 1]

            # Check if edge is forbidden
            if (idx1, idx2) in self.forbidden_edges:
                return False

            # Lazy collision check
            state1 = self.nodes[idx1]
            state2 = self.nodes[idx2]

            if self._check_edge_collision(state1, state2, world_coll):
                # Mark edge as forbidden
                self.forbidden_edges.add((idx1, idx2))
                self.forbidden_edges.add((idx2, idx1))

                # Track edge attempts
                self._update_edge_attempts(idx1)

                return False

        return True

    def _mark_node_collision(self, node_idx: int):
        """Mark a node as being in collision"""
        self.collision_set.add(node_idx)
        # Remove edges to this node
        neighbors = list(self.graph.neighbors(node_idx))
        for neighbor in neighbors:
            self.graph.remove_edge(node_idx, neighbor)

    def _update_edge_attempts(self, node_idx: int):
        """Update edge attempt count and potentially mark node as collision"""
        self.edge_attempt_count[node_idx] = self.edge_attempt_count.get(node_idx, 0) + 1

        if (
            self.edge_attempt_count[node_idx]
            >= self.options.max_num_blocked_edges_before_discard
        ):
            self._mark_node_collision(node_idx)

    def _build_trajectory(
        self,
        start: ConstantCurvatureState,
        goal: ConstantCurvatureState,
        path_indices: List[int],
    ) -> ConstantCurvatureState:
        """Build continuous trajectory from discrete path"""
        trajectory_segments = []

        # Start to first node
        first_node = self.nodes[path_indices[0]]
        start_segment = self._batch_interpolate(
            start, first_node, self.options.edge_interpolation_steps
        )
        trajectory_segments.append(start.repeat(1, axis=0))
        trajectory_segments.append(start_segment)

        # Between nodes
        for i in range(len(path_indices) - 1):
            state1 = self.nodes[path_indices[i]]
            state2 = self.nodes[path_indices[i + 1]]

            segment = self._batch_interpolate(
                state1, state2, self.options.edge_interpolation_steps
            )
            trajectory_segments.append(state1.repeat(1, axis=0))
            trajectory_segments.append(segment)

        # Last node to goal
        last_node = self.nodes[path_indices[-1]]
        goal_segment = self._batch_interpolate(
            last_node, goal, self.options.edge_interpolation_steps
        )
        trajectory_segments.append(last_node.repeat(1, axis=0))
        trajectory_segments.append(goal_segment)
        trajectory_segments.append(goal.repeat(1, axis=0))

        # Concatenate all segments
        full_trajectory = jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0), *trajectory_segments
        )

        return full_trajectory

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

        # Restore nodes
        for node in roadmap_data["nodes"]:
            self.add_node(node)

        # Restore edges
        for u, v, data in roadmap_data["edges"]:
            self.graph.add_edge(u, v, **data)

        # Restore collision tracking
        self.collision_set = roadmap_data["collision_set"]
        self.forbidden_edges = roadmap_data["forbidden_edges"]
