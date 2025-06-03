import jax
import jax.numpy as jnp
from jax import lax


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
