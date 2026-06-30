from __future__ import annotations

from typing import Callable, Dict, Tuple, cast

import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from .geometry import BoundingBox, Capsule, CollGeom, HalfSpace, Sphere
from .geometry_pairs import (
    bounding_box_bounding_box,
    bounding_box_halfspace,
    capsule_bounding_box,
    capsule_capsule,
    halfspace_capsule,
    halfspace_sphere,
    sphere_bounding_box,
    sphere_capsule,
    sphere_sphere,
)

COLLISION_FUNCTIONS: Dict[Tuple[type[CollGeom], type[CollGeom]], Callable[
    ..., Float[Array, "*batch"]]] = {
        (HalfSpace, Sphere): halfspace_sphere,
        (HalfSpace, Capsule): halfspace_capsule,
        (Sphere, Sphere): sphere_sphere,
        (Sphere, Capsule): sphere_capsule,
        (Sphere, BoundingBox): sphere_bounding_box,
        (Capsule, Capsule): capsule_capsule,
        (Capsule, BoundingBox): capsule_bounding_box,
        (BoundingBox, BoundingBox): bounding_box_bounding_box,
        (BoundingBox, HalfSpace): bounding_box_halfspace,
    }


def _get_coll_func(
    geom1_cls: type[CollGeom], geom2_cls: type[CollGeom]
) -> Callable[[CollGeom, CollGeom], Float[Array, "*batch"]]:
    """Get appropriate collision function (distance only) for given types."""
    func = COLLISION_FUNCTIONS.get((geom1_cls, geom2_cls))
    if func is not None:
        return cast(
            Callable[[CollGeom, CollGeom], Float[Array, "*batch"]], func
        )

    func_swapped = COLLISION_FUNCTIONS.get((geom2_cls, geom1_cls))
    if func_swapped is not None:
        return cast(
            Callable[[CollGeom, CollGeom], Float[Array, "*batch"]],
            lambda g1,
            g2: func_swapped(g2, g1),
        )

    raise NotImplementedError(
        f"No collision function found for {geom1_cls.__name__} and "
        f"{geom2_cls.__name__}"
    )


@jdc.jit
def collide(geom1: CollGeom, geom2: CollGeom) -> Float[Array, "*batch"]:
    """Calculate collision distance between two geometric objects.

    Handles broadcasting between the two geometry batch shapes.
    """
    try:
        broadcast_shape = jnp.broadcast_shapes(
            geom1.get_batch_axes(), geom2.get_batch_axes()
        )
    except ValueError as e:
        raise ValueError(
            f"Cannot broadcast geometry shapes {geom1.get_batch_axes()} "
            f"and {geom2.get_batch_axes()}"
        ) from e

    geom1_b = geom1.broadcast_to(*broadcast_shape)
    geom2_b = geom2.broadcast_to(*broadcast_shape)

    geom1_cls = type(geom1)
    geom2_cls = type(geom2)

    func = _get_coll_func(geom1_cls, geom2_cls)

    dist_result = func(geom1_b, geom2_b)

    return dist_result


def colldist_from_sdf(
    _dist: Array,
    activation_dist: Array | float,
) -> Array:
    """
    Convert a signed distance field to a collision distance field,
    based on https://arxiv.org/pdf/2310.17274#page=7.39.

    This function applies a smoothing transformation, useful for converting
    raw distances into values suitable for cost functions in optimization.
    It returns values <= 0, where 0 corresponds to distances >=
    activation_dist, and increasingly negative values for deeper penetrations.

    Args:
        _dist: Signed distance field values (positive = separation,
            negative = penetration).
        activation_dist: The distance threshold (margin) below which the cost
            activates.

    Returns:
        Transformed collision distance field values (<= 0).
    """
    _dist = jnp.minimum(_dist, activation_dist)
    _dist = jnp.where(
        _dist < 0,
        _dist - 0.5 * activation_dist,
        -0.5 / (activation_dist + 1e-6) * (_dist - activation_dist)**2,
    )
    _dist = jnp.minimum(_dist, 0.0)
    return _dist
