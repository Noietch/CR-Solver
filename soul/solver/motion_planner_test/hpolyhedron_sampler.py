"""
HPolyhedron-based sampler for configuration space
"""

import jax
import jax.numpy as jnp
from typing import Optional
from dataclasses import dataclass

from ...robots.cc_robot import CCRobot, ConstantCurvatureState


@dataclass
class HPolyhedron:
    """
    Represents a polyhedron in H-representation (Ax <= b).
    """
    A: jnp.ndarray  # Constraint matrix
    b: jnp.ndarray  # Constraint bounds
    ambient_dimension: int
    
    @classmethod
    def make_box(cls, lower_limits: jnp.ndarray, upper_limits: jnp.ndarray) -> 'HPolyhedron':
        """
        Create a box polyhedron from lower and upper bounds.
        """
        dim = len(lower_limits)
        
        # Create constraint matrix for box
        # Format: -x <= -lower, x <= upper for each dimension
        A = jnp.zeros((2 * dim, dim))
        b = jnp.zeros(2 * dim)
        
        for i in range(dim):
            A = A.at[2 * i, i].set(-1)      # -x_i <= -lower_i
            A = A.at[2 * i + 1, i].set(1)   # x_i <= upper_i
            b = b.at[2 * i].set(-lower_limits[i])
            b = b.at[2 * i + 1].set(upper_limits[i])
        
        return cls(A=A, b=b, ambient_dimension=dim)
    
    def chebyshev_center(self) -> jnp.ndarray:
        """
        Compute the Chebyshev center of the polyhedron.
        For a box, this is simply the center point.
        """
        # For a box polyhedron, the Chebyshev center is the geometric center
        # Extract bounds from constraint representation
        dim = self.ambient_dimension
        lower_bounds = jnp.zeros(dim)
        upper_bounds = jnp.zeros(dim)
        
        for i in range(dim):
            # From -x_i <= -lower_i, we get lower_i = -b[2*i]
            lower_bounds = lower_bounds.at[i].set(-self.b[2 * i])
            # From x_i <= upper_i, we get upper_i = b[2*i+1]
            upper_bounds = upper_bounds.at[i].set(self.b[2 * i + 1])
        
        # Return the center
        return (lower_bounds + upper_bounds) / 2.0
    
    def point_in_set(self, point: jnp.ndarray, tol: float = 1e-6) -> bool:
        """
        Check if a point is inside the polyhedron.
        """
        return jnp.all(jnp.dot(self.A, point) <= self.b + tol)


class HPolyhedronSampler:
    """
    Sampler for robot configuration space using HPolyhedron representation.
    Implements Hit-and-Run sampling algorithm.
    """
    
    def __init__(self, robot: CCRobot, seed: int = 42):
        self.robot = robot
        self.key = jax.random.PRNGKey(seed)
        
        # Build configuration space polyhedron
        self.polyhedron = self._build_configuration_space_polyhedron()
        
        # Get Chebyshev center as initial point
        self.chebyshev_center_point = self._compute_chebyshev_center()
        
        # JIT compile sampling functions (mixing_steps is static)
        self._uniform_sample_fn = jax.jit(self._uniform_sample_step, static_argnums=(1,))
    
    def _build_configuration_space_polyhedron(self) -> HPolyhedron:
        """
        Build the configuration space as a box polyhedron.
        Only includes arm parameters (theta and phi), base is fixed at origin.
        """
        # Convert to arrays and repeat for each section if needed
        theta_lower = jnp.atleast_1d(self.robot.config.lower_limits_theta)
        theta_upper = jnp.atleast_1d(self.robot.config.upper_limits_theta)
        phi_lower = jnp.atleast_1d(self.robot.config.lower_limits_phi)
        phi_upper = jnp.atleast_1d(self.robot.config.upper_limits_phi)
        
        # If single value, repeat for each section
        if theta_lower.shape[0] == 1:
            theta_lower = jnp.repeat(theta_lower, self.robot.config.num_sections)
            theta_upper = jnp.repeat(theta_upper, self.robot.config.num_sections)
        if phi_lower.shape[0] == 1:
            phi_lower = jnp.repeat(phi_lower, self.robot.config.num_sections)
            phi_upper = jnp.repeat(phi_upper, self.robot.config.num_sections)
        
        # Only arm bounds (no base position)
        lower_limits = jnp.concatenate([theta_lower, phi_lower])
        upper_limits = jnp.concatenate([theta_upper, phi_upper])
        
        return HPolyhedron.make_box(lower_limits, upper_limits)
    
    def _compute_chebyshev_center(self) -> ConstantCurvatureState:
        """
        Compute the Chebyshev center in configuration space.
        Base is fixed at origin.
        """
        center_vec = self.polyhedron.chebyshev_center()
        
        # Split into components (only theta and phi)
        base_position = jnp.zeros(3)  # Fixed at origin
        theta = center_vec[:self.robot.config.num_sections]
        phi = center_vec[self.robot.config.num_sections:]
        
        return ConstantCurvatureState(
            base_position=base_position,
            theta=theta,
            phi=phi
        )
    
    def get_feasible_point(self) -> ConstantCurvatureState:
        """
        Get a feasible starting point (Chebyshev center).
        """
        return self.chebyshev_center_point
    
    def uniform_sample(
        self,
        previous_sample: Optional[ConstantCurvatureState] = None,
        mixing_steps: int = 10
    ) -> ConstantCurvatureState:
        """
        Sample uniformly from the configuration space using Hit-and-Run.
        """
        if previous_sample is None:
            previous_sample = self.chebyshev_center_point
        
        # Convert to vector representation
        prev_vec = self._state_to_vector(previous_sample)
        
        # Run Hit-and-Run sampling
        self.key, subkey = jax.random.split(self.key)
        new_vec = self._uniform_sample_fn(prev_vec, mixing_steps, subkey)
        
        # Convert back to state
        return self._vector_to_state(new_vec)
    
    def _uniform_sample_step(
        self,
        previous_sample: jnp.ndarray,
        mixing_steps: int,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Hit-and-Run sampling algorithm implementation.
        """
        def one_step(carry, _):
            current_sample, key = carry
            
            # Generate random Gaussian direction
            key, subkey = jax.random.split(key)
            direction = jax.random.normal(subkey, shape=current_sample.shape)
            
            # Compute valid range along direction
            # line_b = b - A * current_sample
            # line_A = A * direction
            line_b = self.polyhedron.b - jnp.dot(self.polyhedron.A, current_sample)
            line_A = jnp.dot(self.polyhedron.A, direction)
            
            # Find theta_min and theta_max
            theta_mins = jnp.where(
                line_A < 0,
                line_b / (line_A - 1e-10),  # Avoid division by zero
                -1e10  # Large negative number
            )
            theta_maxs = jnp.where(
                line_A > 0,
                line_b / (line_A + 1e-10),  # Avoid division by zero
                1e10  # Large positive number
            )
            
            theta_min = jnp.max(theta_mins)
            theta_max = jnp.min(theta_maxs)
            
            # Sample theta uniformly from [theta_min, theta_max]
            key, subkey = jax.random.split(key)
            theta = jax.random.uniform(subkey, minval=theta_min, maxval=theta_max)
            
            # Update sample
            new_sample = current_sample + theta * direction
            
            return (new_sample, key), None
        
        # Run mixing steps
        (final_sample, _), _ = jax.lax.scan(
            one_step, (previous_sample, key), None, length=mixing_steps
        )
        
        return final_sample
    
    def _state_to_vector(self, state: ConstantCurvatureState) -> jnp.ndarray:
        """Convert ConstantCurvatureState to vector representation.
        Only includes arm parameters, base is fixed.
        """
        return jnp.concatenate([
            state.theta,
            state.phi
        ])
    
    def _vector_to_state(self, vec: jnp.ndarray) -> ConstantCurvatureState:
        """Convert vector to ConstantCurvatureState.
        Base is always fixed at origin.
        """
        base_position = jnp.zeros(3)  # Fixed at origin
        theta = vec[:self.robot.config.num_sections]
        phi = vec[self.robot.config.num_sections:]
        
        return ConstantCurvatureState(
            base_position=base_position,
            theta=theta,
            phi=phi
        )
    
    def batch_sample(
        self,
        num_samples: int,
        mixing_steps: int = 10
    ) -> list[ConstantCurvatureState]:
        """
        Generate multiple samples using Hit-and-Run algorithm.
        """
        samples = []
        current_sample = self.chebyshev_center_point
        
        for _ in range(num_samples):
            current_sample = self.uniform_sample(current_sample, mixing_steps)
            samples.append(current_sample)
        
        return samples