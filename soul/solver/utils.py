import jax
import jax.numpy as jnp
from jax import lax

from ..robots.pcc_robot import PCCRobot, ConstantCurvatureState


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
    robot: PCCRobot, num_states: int, sample_root: float
) -> ConstantCurvatureState:
    theta = robot.config.lower_limits_theta + roberts_sequence(
        num_states, robot.config.num_sections, sample_root
    ) * (robot.config.upper_limits_theta - robot.config.lower_limits_theta)

    phi = robot.config.lower_limits_phi + roberts_sequence(
        num_states, robot.config.num_sections, sample_root
    ) * (robot.config.upper_limits_phi - robot.config.lower_limits_phi)

    states = ConstantCurvatureState(
        base_position=jnp.zeros((num_states, 3)),
        theta=theta * robot.config.opt_mask[3],
        phi=phi * robot.config.opt_mask[3 + robot.config.num_sections],
    )
    return states
