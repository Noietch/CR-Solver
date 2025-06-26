from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Float, Array

from .geometry import HalfSpace, Sphere, Capsule, BoundingBox
from . import utils


# --- HalfSpace Collision Implementations ---


def _halfspace_sphere_dist(
    halfspace_normal: Float[Array, "*batch 3"],
    halfspace_point: Float[Array, "*batch 3"],
    sphere_pos: Float[Array, "*batch 3"],
    sphere_radius: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between a halfspace boundary plane and sphere center, minus radius."""
    dist = (
        jnp.einsum("...i,...i->...", sphere_pos - halfspace_point, halfspace_normal)
        - sphere_radius
    )
    return dist


def halfspace_sphere(halfspace: HalfSpace, sphere: Sphere) -> Float[Array, "*batch"]:
    """Calculates distance between a halfspace and a sphere."""
    dist = _halfspace_sphere_dist(
        halfspace.normal,
        halfspace.pose.translation(),
        sphere.pose.translation(),
        sphere.radius,
    )
    return dist


def halfspace_capsule(halfspace: HalfSpace, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculates distance between halfspace and capsule (closest end)."""
    halfspace_normal = halfspace.normal
    halfspace_point = halfspace.pose.translation()
    cap_center = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.height[..., None] / 2
    dist1 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center + segment_offset, cap_radius
    )
    dist2 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center - segment_offset, cap_radius
    )
    final_dist = jnp.minimum(dist1, dist2)
    return final_dist


# --- Sphere/Capsule Collision Implementations ---


def _sphere_sphere_dist(
    pos1: Float[Array, "*batch 3"],
    radius1: Float[Array, "*batch"],
    pos2: Float[Array, "*batch 3"],
    radius2: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between two spheres."""
    _, dist_center = utils.normalize_with_norm(pos2 - pos1)
    dist = dist_center - (radius1 + radius2)
    return dist


def sphere_sphere(sphere1: Sphere, sphere2: Sphere) -> Float[Array, "*batch"]:
    """Calculate distance between two spheres."""
    dist = _sphere_sphere_dist(
        sphere1.pose.translation(),
        sphere1.radius,
        sphere2.pose.translation(),
        sphere2.radius,
    )
    return dist


def sphere_capsule(sphere: Sphere, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between sphere and capsule."""
    cap_pos = capsule.pose.translation()
    sphere_pos = sphere.pose.translation()
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.height[..., None] / 2
    cap_a = cap_pos - segment_offset
    cap_b = cap_pos + segment_offset
    pt_on_axis = utils.closest_segment_point(cap_a, cap_b, sphere_pos)
    dist = _sphere_sphere_dist(sphere_pos, sphere.radius, pt_on_axis, capsule.radius)
    return dist


def capsule_capsule(capsule1: Capsule, capsule2: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between two capsules."""
    pos1 = capsule1.pose.translation()
    axis1 = capsule1.axis
    length1 = capsule1.height
    radius1 = capsule1.radius
    segment1_offset = axis1 * length1[..., None] / 2
    a1 = pos1 - segment1_offset
    b1 = pos1 + segment1_offset

    pos2 = capsule2.pose.translation()
    axis2 = capsule2.axis
    length2 = capsule2.height
    radius2 = capsule2.radius
    segment2_offset = axis2 * length2[..., None] / 2
    a2 = pos2 - segment2_offset
    b2 = pos2 + segment2_offset

    pt1_on_axis, pt2_on_axis = utils.closest_segment_to_segment_points(a1, b1, a2, b2)
    dist = _sphere_sphere_dist(pt1_on_axis, radius1, pt2_on_axis, radius2)
    return dist


# --- Bounding Box Helper Functions ---


def _point_to_bbox_dist(
    point: Float[Array, "*batch 3"],
    bbox_center: Float[Array, "*batch 3"],
    bbox_extents: Float[Array, "*batch 3"],
) -> Float[Array, "*batch"]:
    """Helper: Calculate distance from a point to a bounding box."""
    # Convert point to bbox local coordinates (centered at origin)
    local_point = point - bbox_center

    # Calculate distance to box surface
    # For each axis, calculate how far outside the box the point is
    # If inside, distance is 0 for that axis
    half_extents = bbox_extents / 2.0
    distances = jnp.maximum(jnp.abs(local_point) - half_extents, 0.0)

    # If point is inside box in all dimensions, return negative distance to closest face
    inside_mask = jnp.all(jnp.abs(local_point) <= half_extents, axis=-1)
    outside_dist = jnp.linalg.norm(distances, axis=-1)

    # When inside, distance is negative of the smallest distance to any face
    inside_dist = -jnp.min(half_extents - jnp.abs(local_point), axis=-1)

    return jnp.where(inside_mask, inside_dist, outside_dist)


def _bbox_bbox_dist(
    center1: Float[Array, "*batch 3"],
    extents1: Float[Array, "*batch 3"],
    center2: Float[Array, "*batch 3"],
    extents2: Float[Array, "*batch 3"],
) -> Float[Array, "*batch"]:
    """Helper: Calculate distance between two bounding boxes."""
    # Calculate separation along each axis
    center_diff = jnp.abs(center1 - center2)
    combined_extents = (extents1 + extents2) / 2.0

    # Distance is zero if boxes overlap on an axis, positive otherwise
    axis_separations = jnp.maximum(center_diff - combined_extents, 0.0)

    # Overall distance is the norm of axis separations
    # If boxes overlap on all axes, they're intersecting (negative distance)
    overlapping_mask = jnp.all(center_diff <= combined_extents, axis=-1)
    outside_dist = jnp.linalg.norm(axis_separations, axis=-1)

    # When overlapping, return negative of the smallest overlap
    overlaps = combined_extents - center_diff
    inside_dist = -jnp.min(overlaps, axis=-1)

    return jnp.where(overlapping_mask, inside_dist, outside_dist)


# --- Main Collision Functions ---


def capsule_bounding_box(capsule: Capsule, bbox: BoundingBox) -> Float[Array, "*batch"]:
    """Calculate distance between capsule and bounding box."""
    cap_center = capsule.pose.translation()
    cap_axis = capsule.axis
    cap_radius = capsule.radius
    cap_height = capsule.height

    bbox_center = bbox.center
    bbox_extents = bbox.extents

    # Calculate the endpoints of the capsule's central axis
    segment_offset = cap_axis * cap_height[..., None] / 2
    cap_a = cap_center - segment_offset
    cap_b = cap_center + segment_offset

    # Find the point on the capsule axis closest to the bounding box center
    # This gives us a good approximation of the closest approach
    closest_point_on_axis = utils.closest_segment_point(cap_a, cap_b, bbox_center)

    # Calculate distance from this point to the bounding box
    axis_to_bbox_dist = _point_to_bbox_dist(
        closest_point_on_axis, bbox_center, bbox_extents
    )

    # Subtract the capsule radius to get the surface-to-surface distance
    return axis_to_bbox_dist - cap_radius


def bounding_box_bounding_box(
    bbox1: BoundingBox, bbox2: BoundingBox
) -> Float[Array, "*batch"]:
    """Calculate distance between two bounding boxes."""
    return _bbox_bbox_dist(
        bbox1.center,
        bbox1.extents,
        bbox2.center,
        bbox2.extents,
    )


def sphere_bounding_box(sphere: Sphere, bbox: BoundingBox) -> Float[Array, "*batch"]:
    """Calculate distance between sphere and bounding box."""
    sphere_center = sphere.pose.translation()
    sphere_radius = sphere.radius
    bbox_center = bbox.center
    bbox_extents = bbox.extents

    # Calculate distance from sphere center to bounding box
    point_to_box_dist = _point_to_bbox_dist(sphere_center, bbox_center, bbox_extents)

    # Subtract sphere radius to get surface-to-surface distance
    return point_to_box_dist - sphere_radius


def bounding_box_halfspace(
    bbox: BoundingBox, halfspace: HalfSpace
) -> Float[Array, "*batch"]:
    """Calculate distance between bounding box and halfspace."""
    bbox_center = bbox.center
    bbox_extents = bbox.extents
    halfspace_normal = halfspace.normal
    halfspace_point = halfspace.pose.translation()

    # Find the vertex of the bounding box that is furthest in the direction of the halfspace normal
    # This determines the closest approach of the box to the halfspace
    half_extents = bbox_extents / 2.0

    # Project half-extents onto the normal direction
    # Choose sign to get the vertex closest to the halfspace
    projected_extent = jnp.sum(
        jnp.abs(half_extents[..., None] * halfspace_normal), axis=-1
    )

    # Distance from box center to halfspace plane
    center_to_plane_dist = jnp.einsum(
        "...i,...i->...", bbox_center - halfspace_point, halfspace_normal
    )

    # The closest point on the box to the halfspace is center minus projected extent
    return center_to_plane_dist - projected_extent
