import jax
import jax.numpy as jnp
from jax import lax

from ..robots.cc_robot import CCRobot, ConstantCurvatureState
from ..robots.cc_robot_extend import CCRobot as CCRobotExtend
from ..robots.cc_robot_extend import (
    ConstantCurvatureState as ConstantCurvatureStateExtend,
)


def newton_raphson(f, x, iters):
    """Use the Newton-Raphson method to find a root of the given function."""

    def update(x, _):
        y = x - f(x) / jax.grad(f)(x)
        return y, None

    x, _ = lax.scan(update, 1.0, length=iters)
    return x


def roberts_sequence(num_points, dim, root):
    # From https://gist.github.com/carlosgmartin/1fd4e60bed526ec8ae076137ded6ebab.
    basis = 1 - (1 / root ** (1 + jnp.arange(dim)))

    n = jnp.arange(num_points)
    x = n[:, None] * basis[None, :]
    x, _ = jnp.modf(x)

    return x


def sample_states(
    robot: CCRobot, num_states: int, sample_root: float
) -> ConstantCurvatureState:
    theta = robot.config.lower_limits_theta + roberts_sequence(
        num_states, robot.config.num_sections, sample_root
    ) * (robot.config.upper_limits_theta - robot.config.lower_limits_theta)

    phi = robot.config.lower_limits_phi + roberts_sequence(
        num_states, robot.config.num_sections, sample_root
    ) * (robot.config.upper_limits_phi - robot.config.lower_limits_phi)

    if isinstance(robot, CCRobotExtend):
        length = robot.config.lower_limits_length + roberts_sequence(
            num_states, robot.config.num_sections, sample_root
        ) * (robot.config.upper_limits_length - robot.config.lower_limits_length)

        states = ConstantCurvatureStateExtend(
            base_position=jnp.zeros((num_states, 3)),
            theta=theta * robot.config.opt_mask[3],
            phi=phi * robot.config.opt_mask[3 + robot.config.num_sections],
            length=length * robot.config.opt_mask[3 + robot.config.num_sections * 2],
        )
    else:
        states = ConstantCurvatureState(
            base_position=jnp.zeros((num_states, 3)),
            theta=theta * robot.config.opt_mask[3],
            phi=phi * robot.config.opt_mask[3 + robot.config.num_sections],
        )
    return states


def sample_around(
    state: ConstantCurvatureState,
    stddev: float,
    num_samples: int,
    robot: CCRobot,
    key: jax.Array,
):
    theta_noise = stddev * jax.random.normal(
        key, shape=(num_samples, state.theta.shape[0])
    )
    phi_noise = stddev * jax.random.normal(key, shape=(num_samples, state.phi.shape[0]))

    theta = state.theta[None, :] + theta_noise
    phi = state.phi[None, :] + phi_noise

    # Clip to joint limits
    theta = jnp.clip(
        theta, robot.config.lower_limits_theta, robot.config.upper_limits_theta
    )
    phi = jnp.clip(phi, robot.config.lower_limits_phi, robot.config.upper_limits_phi)

    base_position = jnp.tile(state.base_position, (num_samples, 1))  # shape: (N, 3)

    return ConstantCurvatureState(
        base_position=base_position,
        theta=theta * robot.config.opt_mask[3],
        phi=phi * robot.config.opt_mask[3 + robot.config.num_sections],
    )


def sample_states_around_start_goal(
    start: ConstantCurvatureState,
    goal: ConstantCurvatureState,
    stddev: float,
    num_samples: int,
    robot: CCRobot,
    key: jax.Array,
) -> ConstantCurvatureState:
    """
    Gaussian sampling around start and goal, with joint limits enforced.

    Args:
        start, goal: start & goal states
        stddev: standard deviation of Gaussian noise
        num_samples: number of samples per state
        robot: to access joint limits
        key: PRNG key

    Returns:
        Batched ConstantCurvatureState
    """
    key_start, key_goal = jax.random.split(key)

    sampled_from_start = sample_around(start, stddev, num_samples, robot, key_start)
    sampled_from_goal = sample_around(goal, stddev, num_samples, robot, key_goal)

    return ConstantCurvatureState(
        base_position=jnp.concatenate(
            [sampled_from_start.base_position, sampled_from_goal.base_position]
        ),
        theta=jnp.concatenate([sampled_from_start.theta, sampled_from_goal.theta]),
        phi=jnp.concatenate([sampled_from_start.phi, sampled_from_goal.phi]),
    )
