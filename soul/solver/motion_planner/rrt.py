"""
Optimized Rapidly-exploring Random Tree (RRT) Motion Planner for CC Robot
Enhanced JAX-based implementation with parallel operations
"""

from typing import Optional, Sequence, Dict, List
import jax
import jax.numpy as jnp
import jax.tree_util
import networkx as nx
from dataclasses import dataclass

from ...robots.cc_robot import CCRobot, ConstantCurvatureState
from ...geom import RobotCollision, CollGeom
from .utils import sample_around


@dataclass
class RRTOptions:
    """Options for RRT planner"""

    goal_sample_rate: float = 0.2
    step_size: float = 0.1
    max_iterations: int = 1000
    edge_interpolation_steps: int = 10
    distance_tolerance: float = 0.1
    batch_size: int = 100  # For parallel sampling
    use_simple_distance: bool = (
        True  # Use simple L2 distance instead of collision-based
    )


class OptimizedRRT:
    """
    Optimized RRT planner using JAX features.
    Key optimizations:
    - Vectorized collision checking
    - JIT compiled distance computations
    - Efficient goal biased sampling
    - Batch operations for node expansions
    """

    def __init__(
        self,
        robot: CCRobot,
        robot_coll: RobotCollision,
        options: Optional[RRTOptions] = None,
    ):
        self.robot = robot
        self.robot_coll = robot_coll
        self.options = options or RRTOptions()

        # Graph structure (directed tree)
        self.graph = nx.DiGraph()
        self.nodes: List[ConstantCurvatureState] = []
        self.parent: Dict[int, int] = {}  # child_idx -> parent_idx

        # Pre-allocate arrays for nearest neighbor search
        self.nodes_array = None  # Will be updated when nodes are added

        # JIT compiled functions - only compile what's frequently used
        self._simple_distance = jax.jit(self._simple_distance_fn)
        self._check_single_collision = jax.jit(self._check_single_collision_fn)
        self._steer = jax.jit(self._steer_fn)
        self._interpolate_path = jax.jit(self._interpolate_path_fn, static_argnums=(2,))

        # Random key
        self.key = jax.random.PRNGKey(42)

    def clear(self):
        """Clear the RRT tree"""
        self.graph.clear()
        self.nodes.clear()
        self.parent.clear()

    def add_node(
        self, state: ConstantCurvatureState, parent_idx: Optional[int] = None
    ) -> int:
        """Add a node to the tree"""
        idx = len(self.nodes)
        self.nodes.append(state)
        self.graph.add_node(idx, state=state)

        if parent_idx is not None:
            self.parent[idx] = parent_idx
            self.graph.add_edge(parent_idx, idx)

        return idx

    def _check_single_collision_fn(
        self, state: ConstantCurvatureState, world_coll: Sequence[CollGeom]
    ) -> jnp.ndarray:
        """Check collision for a single state"""

        # Check collision with world objects
        has_collision = jnp.array(False)
        for world_obj in world_coll:
            dist_matrix = self.robot_coll.compute_world_collision_distance(
                self.robot, state, world_obj, 1
            )
            has_collision = has_collision | jnp.any(dist_matrix < 0.0)

        return has_collision

    def _batch_check_collision_fn(
        self, states: ConstantCurvatureState, world_coll: Sequence[CollGeom]
    ) -> jnp.ndarray:
        """Vectorized collision checking for multiple states"""

        # Check for NaN values
        nan_mask = jnp.any(jnp.isnan(states.theta), axis=1) | jnp.any(
            jnp.isnan(states.phi), axis=1
        )

        # Vectorized collision check for all world objects
        def check_single_object(world_obj):
            dist_matrix = jax.vmap(
                lambda s: self.robot_coll.compute_world_collision_distance(
                    self.robot, s, world_obj, 1
                )
            )(states)
            return jnp.any(dist_matrix < 0.0, axis=(1, 2))

        # Stack collision results
        if world_coll:
            collision_masks = jnp.stack(
                [check_single_object(obj) for obj in world_coll]
            )
            world_collision = jnp.any(collision_masks, axis=0)
        else:
            world_collision = jnp.zeros(states.theta.shape[0], dtype=bool)

        return nan_mask | world_collision

    def _simple_distance_fn(
        self, state1: ConstantCurvatureState, state2: ConstantCurvatureState
    ) -> jnp.ndarray:
        """Simple L2 distance computation between two states"""
        theta_dist = jnp.sum((state2.theta - state1.theta) ** 2)
        phi_dist = jnp.sum((state2.phi - state1.phi) ** 2)
        base_dist = jnp.sum((state2.base_position - state1.base_position) ** 2)

        return jnp.sqrt(theta_dist + phi_dist + base_dist)

    def _batch_distance_fn(
        self, state1: ConstantCurvatureState, states2: ConstantCurvatureState
    ) -> jnp.ndarray:
        """Vectorized distance computation"""
        theta_dist = jnp.sum((states2.theta - state1.theta[None, :]) ** 2, axis=1)
        phi_dist = jnp.sum((states2.phi - state1.phi[None, :]) ** 2, axis=1)
        base_dist = jnp.sum(
            (states2.base_position - state1.base_position[None, :]) ** 2, axis=1
        )

        return jnp.sqrt(theta_dist + phi_dist + base_dist)

    def _steer_fn(
        self, from_state: ConstantCurvatureState, to_state: ConstantCurvatureState
    ) -> ConstantCurvatureState:
        """Steer from one state towards another with step size limit"""
        # Compute distance
        theta_dist = jnp.sum((to_state.theta - from_state.theta) ** 2)
        phi_dist = jnp.sum((to_state.phi - from_state.phi) ** 2)
        base_dist = jnp.sum((to_state.base_position - from_state.base_position) ** 2)
        total_dist = jnp.sqrt(theta_dist + phi_dist + base_dist)

        # Compute interpolation factor
        alpha = jnp.minimum(self.options.step_size / (total_dist + 1e-6), 1.0)

        # Interpolate
        new_base = (
            1 - alpha
        ) * from_state.base_position + alpha * to_state.base_position
        new_theta = (1 - alpha) * from_state.theta + alpha * to_state.theta
        new_phi = (1 - alpha) * from_state.phi + alpha * to_state.phi

        return ConstantCurvatureState(
            base_position=new_base, theta=new_theta, phi=new_phi
        )

    def _batch_steer_fn(
        self, from_states: ConstantCurvatureState, to_states: ConstantCurvatureState
    ) -> ConstantCurvatureState:
        """Vectorized steering for multiple state pairs"""
        # Compute distances
        theta_dist = jnp.sum((to_states.theta - from_states.theta) ** 2, axis=1)
        phi_dist = jnp.sum((to_states.phi - from_states.phi) ** 2, axis=1)
        base_dist = jnp.sum(
            (to_states.base_position - from_states.base_position) ** 2, axis=1
        )
        total_dist = jnp.sqrt(theta_dist + phi_dist + base_dist)

        # Compute interpolation factors
        alpha = jnp.minimum(self.options.step_size / (total_dist + 1e-6), 1.0)[:, None]

        # Vectorized interpolation
        new_base = (
            1 - alpha
        ) * from_states.base_position + alpha * to_states.base_position
        new_theta = (1 - alpha) * from_states.theta + alpha * to_states.theta
        new_phi = (1 - alpha) * from_states.phi + alpha * to_states.phi

        return ConstantCurvatureState(
            base_position=new_base, theta=new_theta, phi=new_phi
        )

    def _interpolate_path_fn(
        self,
        state1: ConstantCurvatureState,
        state2: ConstantCurvatureState,
        num_steps: int,
    ) -> ConstantCurvatureState:
        """Interpolate between two states"""
        alphas = jnp.linspace(0, 1, num_steps + 2)[1:-1]

        # Vectorized interpolation
        base_positions = jnp.outer(1 - alphas, state1.base_position) + jnp.outer(
            alphas, state2.base_position
        )
        thetas = jnp.outer(1 - alphas, state1.theta) + jnp.outer(alphas, state2.theta)
        phis = jnp.outer(1 - alphas, state1.phi) + jnp.outer(alphas, state2.phi)

        return ConstantCurvatureState(
            base_position=base_positions, theta=thetas, phi=phis
        )

    def sample_goal_biased(
        self, goal: ConstantCurvatureState, num_samples: int = 1
    ) -> ConstantCurvatureState:
        """Sample with goal bias"""
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)

        # Determine which samples should be the goal
        goal_mask = (
            jax.random.uniform(subkey1, (num_samples,)) < self.options.goal_sample_rate
        )

        # Sample around goal for non-goal samples
        sampled = sample_around(goal, 0.5, num_samples, self.robot, subkey2)

        # Mix goal and random samples
        base_positions = jnp.where(
            goal_mask[:, None],
            jnp.tile(goal.base_position[None, :], (num_samples, 1)),
            sampled.base_position,
        )
        thetas = jnp.where(
            goal_mask[:, None],
            jnp.tile(goal.theta[None, :], (num_samples, 1)),
            sampled.theta,
        )
        phis = jnp.where(
            goal_mask[:, None],
            jnp.tile(goal.phi[None, :], (num_samples, 1)),
            sampled.phi,
        )

        return ConstantCurvatureState(
            base_position=base_positions, theta=thetas, phi=phis
        )

    def find_nearest_node(self, query: ConstantCurvatureState) -> int:
        """Find nearest node in the tree using simple L2 distance"""
        if not self.nodes:
            return -1

        # Use simple distance for faster computation
        distances = jnp.array(
            [self._simple_distance(query, node) for node in self.nodes]
        )
        return int(jnp.argmin(distances))

    def check_edge_collision(
        self,
        state1: ConstantCurvatureState,
        state2: ConstantCurvatureState,
        world_coll: Sequence[CollGeom],
    ) -> bool:
        """Check if edge between two states is collision-free"""
        # Use fewer interpolation steps for edge checking during search
        num_steps = min(
            5, self.options.edge_interpolation_steps
        )  # Reduce to 5 for speed

        # Manual interpolation to avoid overhead
        for i in range(1, num_steps + 1):
            alpha = i / (num_steps + 1)
            interp_base = (
                1 - alpha
            ) * state1.base_position + alpha * state2.base_position
            interp_theta = (1 - alpha) * state1.theta + alpha * state2.theta
            interp_phi = (1 - alpha) * state1.phi + alpha * state2.phi

            interp_state = ConstantCurvatureState(
                base_position=interp_base, theta=interp_theta, phi=interp_phi
            )

            if bool(self._check_single_collision(interp_state, world_coll)):
                return False

        return True

    def is_goal_reached(
        self, state: ConstantCurvatureState, goal: ConstantCurvatureState
    ) -> bool:
        """Check if state is close enough to goal"""
        base_close = jnp.allclose(
            state.base_position,
            goal.base_position,
            atol=self.options.distance_tolerance,
        )
        theta_close = jnp.allclose(
            state.theta, goal.theta, atol=self.options.distance_tolerance
        )
        phi_close = jnp.allclose(
            state.phi, goal.phi, atol=self.options.distance_tolerance
        )

        return base_close and theta_close and phi_close

    def reconstruct_path(self, start_idx: int, goal_idx: int) -> ConstantCurvatureState:
        """Reconstruct path from start to goal"""
        # Build path indices
        path_indices = []
        current = goal_idx

        while current != start_idx:
            path_indices.append(current)
            current = self.parent[current]
        path_indices.append(start_idx)
        path_indices.reverse()

        # Get path nodes
        path_nodes = [self.nodes[idx] for idx in path_indices]

        # Interpolate full trajectory
        trajectory_segments = []

        for i in range(len(path_nodes)):
            # Add current node
            trajectory_segments.append(
                jax.tree_util.tree_map(lambda x: x[None, ...], path_nodes[i])
            )

            # Interpolate to next node if not last
            if i < len(path_nodes) - 1:
                segment = self._interpolate_path(
                    path_nodes[i],
                    path_nodes[i + 1],
                    self.options.edge_interpolation_steps,
                )
                trajectory_segments.append(segment)

        # Concatenate all segments
        full_trajectory = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), *trajectory_segments
        )

        return full_trajectory

    def find_path(
        self,
        start: ConstantCurvatureState,
        goal: ConstantCurvatureState,
        world_coll: Sequence[CollGeom],
    ) -> Optional[ConstantCurvatureState]:
        """
        Find path from start to goal using RRT.
        """
        # Check start and goal for collision
        if bool(self._check_single_collision(start, world_coll)):
            print("Start configuration is in collision")
            return None
        if bool(self._check_single_collision(goal, world_coll)):
            print("Goal configuration is in collision")
            return None

        # Clear and initialize tree
        self.clear()
        self.add_node(start)

        # RRT main loop
        for iteration in range(self.options.max_iterations):
            # Sample (with goal bias)
            sample = self.sample_goal_biased(goal, 1)
            sample_single = jax.tree_util.tree_map(lambda x: x[0], sample)

            # Find nearest node
            nearest_idx = self.find_nearest_node(sample_single)
            if nearest_idx < 0:
                continue

            nearest = self.nodes[nearest_idx]

            # Steer towards sample
            new_state = self._steer(nearest, sample_single)

            # Check collision for new state
            if bool(self._check_single_collision(new_state, world_coll)):
                continue

            # Check edge collision
            if not self.check_edge_collision(nearest, new_state, world_coll):
                continue

            # Add new node to tree
            new_idx = self.add_node(new_state, nearest_idx)

            # Check if goal is reached
            if self.is_goal_reached(new_state, goal):
                # Try to connect to goal
                if self.check_edge_collision(new_state, goal, world_coll):
                    goal_idx = self.add_node(goal, new_idx)
                    print(f"Path found after {iteration + 1} iterations")
                    return self.reconstruct_path(0, goal_idx)

            # Progress report
            if (iteration + 1) % 100 == 0:
                print(f"RRT iteration {iteration + 1}/{self.options.max_iterations}")

        # If no exact path found, find nearest node to goal
        print("Max iterations reached, finding approximate path...")
        return self._find_approximate_path(goal, world_coll)

    def _find_approximate_path(
        self, goal: ConstantCurvatureState, world_coll: Sequence[CollGeom]
    ) -> Optional[ConstantCurvatureState]:
        """Find approximate path when exact path not found"""
        if not self.nodes:
            return None

        # Find nearest node to goal
        nearest_idx = self.find_nearest_node(goal)
        if nearest_idx < 0:
            return None

        nearest = self.nodes[nearest_idx]

        # Get path to nearest node
        path = self.reconstruct_path(0, nearest_idx)

        # Try to extend to goal
        if self.check_edge_collision(nearest, goal, world_coll):
            # Can connect to goal
            final_segment = self._interpolate_path(
                nearest, goal, self.options.edge_interpolation_steps
            )

            # Concatenate path and final segment
            full_path = jax.tree_util.tree_map(
                lambda p, s: jnp.concatenate([p, s], axis=0), path, final_segment
            )

            # Add goal at the end
            goal_expanded = jax.tree_util.tree_map(lambda x: x[None, ...], goal)
            full_path = jax.tree_util.tree_map(
                lambda p, g: jnp.concatenate([p, g], axis=0), full_path, goal_expanded
            )

            print(
                f"Approximate path found (distance to goal: {self._simple_distance(nearest, goal):.4f})"
            )
            return full_path
        else:
            print(f"Cannot connect to goal, returning path to nearest point")
            return path
