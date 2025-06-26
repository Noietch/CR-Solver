from __future__ import annotations

import os
import pickle
import coacd
import trimesh
import jaxlie
import jax.numpy as jnp
from typing import Tuple
from jaxtyping import Float, Array
from trimesh.voxel import creation

_SAFE_EPS = 1e-6


def _apply_transform_to_mesh(
    mesh: trimesh.Trimesh, scale: float, wxyz: Array | list | None, position: Array | list | None
) -> None:
    """Apply scale and transform to a mesh in-place."""
    if scale != 1.0:
        # Scale around the mesh center to preserve position
        mesh.apply_scale(scale)

    if wxyz is not None:
        if position is None:
            position = jnp.array([0.0, 0.0, 0.0])
        mesh.apply_transform(
            jaxlie.SE3.from_rotation_and_translation(
                rotation=jaxlie.SO3(wxyz=jnp.array(wxyz)),
                translation=jnp.array(position),
            ).as_matrix()
        )


def load_mesh(
    path: str,
    scale: float = 1.0,
    wxyz: Array | list | None = None,
    position: Array | list | None = None,
    decompose_type: str | None = None,
    decompose_params: dict | None = None,
) -> trimesh.Trimesh | list[trimesh.Trimesh]:
    mesh = trimesh.load(path, force="mesh")
    
    if decompose_type is None:
        _apply_transform_to_mesh(mesh, scale, wxyz)
        return mesh, mesh

    # Handle convex decomposition
    if decompose_type == "convex":
        max_convex_hull = decompose_params.get("max_convex_hull", 10)
        parts_path = path + f"_convex_parts_{max_convex_hull}.pkl"
        if os.path.exists(parts_path):
            with open(parts_path, "rb") as f:
                parts = pickle.load(f)
        else:
            mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)
            parts = coacd.run_coacd(mesh_coacd, max_convex_hull=max_convex_hull)
            with open(parts_path, "wb") as f:
                pickle.dump(parts, f)
        meshes = [trimesh.Trimesh(part[0], part[1]) for part in parts]
    elif decompose_type == "voxel":
        pitch = decompose_params.get("pitch", 0.4)
        voxel_path = path + f"_voxel_parts_{pitch}.pkl"
        if os.path.exists(voxel_path):
            with open(voxel_path, "rb") as f:
                meshes = pickle.load(f)
        else:
            voxel_grid = creation.voxelize(mesh, pitch=pitch)
            meshes = []
            for center_point in voxel_grid.fill().points:
                single_cube = trimesh.creation.box()
                single_cube.apply_scale(voxel_grid.pitch)
                single_cube.apply_translation(center_point)
                meshes.append(single_cube)
            with open(voxel_path, "wb") as f:
                pickle.dump(meshes, f)
    else:
        raise ValueError(f"Unknown decompose type: {decompose_type}")

    for m in meshes:
        _apply_transform_to_mesh(m, scale, wxyz, position)
    _apply_transform_to_mesh(mesh, scale, wxyz, position)
    print(f"Loaded {len(meshes)} convex parts from {path}")
    return meshes, mesh 


def make_frame(direction: Array) -> Array:
    """Make a frame from a direction vector, aligning the z-axis with the direction."""
    # Based on `mujoco.mjx._src.math.make_frame`.

    is_zero = jnp.isclose(direction, 0.0).all(axis=-1, keepdims=True)
    direction = jnp.where(
        is_zero,
        jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), direction.shape),
        direction,
    )
    direction /= jnp.linalg.norm(direction, axis=-1, keepdims=True) + _SAFE_EPS

    y = jnp.broadcast_to(jnp.array([0, 1, 0]), (*direction.shape[:-1], 3))
    z = jnp.broadcast_to(jnp.array([0, 0, 1]), (*direction.shape[:-1], 3))

    normal = jnp.where((-0.5 < direction[..., 1:2]) & (direction[..., 1:2] < 0.5), y, z)
    normal -= direction * jnp.einsum("...i,...i->...", normal, direction)[..., None]
    normal /= jnp.linalg.norm(normal, axis=-1, keepdims=True) + _SAFE_EPS

    return jnp.stack([jnp.cross(normal, direction), normal, direction], axis=-1)


def normalize(x: Float[Array, "*batch N"]) -> Float[Array, "*batch N"]:
    """Normalizes a vector, handling the zero vector."""
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    safe_norm = jnp.where(norm == 0.0, 1.0, norm)
    normalized_x = x / safe_norm
    return jnp.where(norm == 0.0, jnp.zeros_like(x), normalized_x)


def normalize_with_norm(
    x: Float[Array, "*batch N"],
) -> Tuple[Float[Array, "*batch N"], Float[Array, "*batch"]]:
    """Normalizes a vector and returns the norm, handling the zero vector."""
    norm = jnp.linalg.norm(x + 1e-6, axis=-1, keepdims=True)
    safe_norm = jnp.where(norm == 0.0, 1.0, norm)
    normalized_x = x / safe_norm
    result_vec = jnp.where(norm == 0.0, jnp.zeros_like(x), normalized_x)
    result_norm = norm[..., 0]
    return result_vec, result_norm


def closest_segment_point(
    a: Float[Array, "*batch 3"],
    b: Float[Array, "*batch 3"],
    pt: Float[Array, "*batch 3"],
) -> Float[Array, "*batch 3"]:
    """Finds the closest point on the line segment [a, b] to point pt."""
    ab = b - a
    t = jnp.einsum("...i,...i->...", pt - a, ab) / (
        jnp.einsum("...i,...i->...", ab, ab) + _SAFE_EPS
    )
    t_clamped = jnp.clip(t, 0.0, 1.0)
    return a + ab * t_clamped[..., None]


def closest_segment_to_segment_points(
    a1: Float[Array, "*batch 3"],
    b1: Float[Array, "*batch 3"],
    a2: Float[Array, "*batch 3"],
    b2: Float[Array, "*batch 3"],
) -> Tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
    """Finds the closest points between two line segments [a1, b1] and [a2, b2]."""
    d1 = b1 - a1  # Direction vector of segment S1
    d2 = b2 - a2  # Direction vector of segment S2
    r = a1 - a2

    a = jnp.einsum("...i,...i->...", d1, d1)  # Squared length of segment S1
    e = jnp.einsum("...i,...i->...", d2, d2)  # Squared length of segment S2
    f = jnp.einsum("...i,...i->...", d2, r)
    c = jnp.einsum("...i,...i->...", d1, r)
    b = jnp.einsum("...i,...i->...", d1, d2)
    denom = a * e - b * b  # Squared area of the parallelogram defined by d1, d2

    s_num = b * f - c * e
    t_num = a * f - b * c

    s_parallel = -c / (a + _SAFE_EPS)
    t_parallel = f / (e + _SAFE_EPS)

    s = jnp.where(denom < _SAFE_EPS, s_parallel, s_num / (denom + _SAFE_EPS))
    t = jnp.where(denom < _SAFE_EPS, t_parallel, t_num / (denom + _SAFE_EPS))

    s_clamped = jnp.clip(s, 0.0, 1.0)
    t_clamped = jnp.clip(t, 0.0, 1.0)

    t_recomp = jnp.einsum(
        "...i,...i->...", d2, (a1 + d1 * s_clamped[..., None]) - a2
    ) / (e + _SAFE_EPS)
    t_final = jnp.where(
        jnp.abs(s - s_clamped) > _SAFE_EPS, jnp.clip(t_recomp, 0.0, 1.0), t_clamped
    )

    s_recomp = jnp.einsum("...i,...i->...", d1, (a2 + d2 * t_final[..., None]) - a1) / (
        a + _SAFE_EPS
    )
    s_final = jnp.where(
        jnp.abs(t - t_final) > _SAFE_EPS, jnp.clip(s_recomp, 0.0, 1.0), s_clamped
    )

    c1 = a1 + d1 * s_final[..., None]
    c2 = a2 + d2 * t_final[..., None]
    return c1, c2
