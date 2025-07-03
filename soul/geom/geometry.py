from __future__ import annotations

import abc
from typing import cast, Self

import trimesh

import jax.numpy as jnp
import jaxlie
from jaxtyping import Float, Array
import jax_dataclasses as jdc
import numpy as onp
import jax
import jax.scipy.ndimage

from .utils import make_frame


def cat_geoms(geoms: list[CollGeom]) -> CollGeom:
    """Concatenate a list of geometries into a single geometry.

    This function handles both single geometries and batched geometries.
    All input geometries are flattened and then concatenated along the first dimension.

    Args:
        geoms: List of CollGeom objects. Each can be single or batched.

    Returns:
        A single CollGeom object containing all input geometries in its batch dimension.
    """
    if not geoms:
        raise ValueError("Cannot concatenate empty list of geometries")

    # Check that all geometries are of the same type
    first_type = type(geoms[0])
    if not all(isinstance(g, first_type) for g in geoms):
        raise TypeError("All geometries must be of the same type")

    # Flatten each geometry and collect poses and sizes
    all_poses = []
    all_sizes = []

    for geom in geoms:
        batch_axes = geom.get_batch_axes()
        if batch_axes:
            # Flatten the batched geometry
            flat_pose = geom.pose.wxyz_xyz.reshape(-1, 7)
            flat_size = geom.size.reshape(-1, geom.size.shape[-1])
        else:
            # Single geometry - add batch dimension
            flat_pose = geom.pose.wxyz_xyz[None, :]
            flat_size = geom.size[None, :]

        all_poses.append(flat_pose)
        all_sizes.append(flat_size)

    # Concatenate all poses and sizes
    combined_poses = jnp.concatenate(all_poses, axis=0)
    combined_sizes = jnp.concatenate(all_sizes, axis=0)

    return first_type(pose=jaxlie.SE3(wxyz_xyz=combined_poses), size=combined_sizes)


@jdc.pytree_dataclass
class CollGeom(abc.ABC):
    """Base class for geometric objects."""

    pose: jaxlie.SE3
    """Geometry pose (position and orientation)."""

    size: Float[Array, "*batch shape_dim"]
    """Geometry shape parameters, e.g. radius, half-length."""

    def get_batch_axes(self) -> tuple[int, ...]:
        """Get batch axes of the geometry."""
        batch_axes_from_pose = self.pose.get_batch_axes()
        size_batch_axes = self.size.shape[:-1]
        assert (
            size_batch_axes == batch_axes_from_pose
        ), f"Size batch axes {size_batch_axes} do not match pose batch axes {batch_axes_from_pose}."
        return batch_axes_from_pose

    def broadcast_to(self, *shape: int) -> Self:
        """Broadcast geometry to given shape."""
        new_pose_wxyz_xyz = jnp.broadcast_to(self.pose.wxyz_xyz, shape + (7,))
        new_pose = jaxlie.SE3(new_pose_wxyz_xyz)
        shape_dim = self.size.shape[-1]
        new_size = jnp.broadcast_to(self.size, shape + (shape_dim,))
        return type(self)(pose=new_pose, size=new_size)

    def reshape(self, *shape: int) -> Self:
        """Reshape geometry to given shape."""
        new_pose_wxyz_xyz = self.pose.wxyz_xyz.reshape(shape + (7,))
        new_pose = jaxlie.SE3(new_pose_wxyz_xyz)
        shape_dim = self.size.shape[-1]
        new_size = self.size.reshape(shape + (shape_dim,))
        return type(self)(pose=new_pose, size=new_size)

    def set_transform(self, transform: jaxlie.SE3) -> Self:
        new_pose = transform
        new_batch_axes = new_pose.get_batch_axes()
        broadcast_size = jnp.broadcast_to(
            self.size, new_batch_axes + self.size.shape[-1:]
        )
        kwargs = {"pose": new_pose, "size": broadcast_size}
        return type(self)(**kwargs)

    def transform(self, transform: jaxlie.SE3) -> Self:
        """Applies an SE3 transformation to the geometry."""
        new_pose = transform @ self.pose
        new_batch_axes = new_pose.get_batch_axes()
        broadcast_size = jnp.broadcast_to(
            self.size, new_batch_axes + self.size.shape[-1:]
        )
        kwargs = {"pose": new_pose, "size": broadcast_size}
        return type(self)(**kwargs)

    def transform_from_pos_wxyz(
        self,
        position: Float[Array, "*batch 3"],
        wxyz: Float[Array, "*batch 4"],
    ) -> Self:
        """
        Transform the geometry from a position and orientation.

        Equivalent to `self.transform`, but doesn't require direct JAX instantiation of SE3.
        """
        position, wxyz = jnp.array(position), jnp.array(wxyz)
        pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), position)
        return self.transform(pose)

    @abc.abstractmethod
    def _create_one_mesh(self, index: tuple[int, ...]) -> trimesh.Trimesh:
        """Helper to create a single trimesh object from batch data at a given index."""
        raise NotImplementedError

    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert the (potentially batched) geometry to a single trimesh object."""
        batch_axes = self.get_batch_axes()
        if not batch_axes:
            return self._create_one_mesh(tuple())

        meshes = [
            self._create_one_mesh(idx_tuple) for idx_tuple in onp.ndindex(batch_axes)
        ]
        if not meshes:
            return trimesh.Trimesh()

        return cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))


@jdc.pytree_dataclass
class HalfSpace(CollGeom):
    """HalfSpace geometry defined by a point and an outward normal."""

    @property
    def normal(self) -> Float[Array, "*batch 3"]:
        """Normal vector (Z-axis of rotation matrix)."""
        return self.pose.rotation().as_matrix()[..., :, 2]

    @property
    def offset(self) -> Float[Array, "*batch"]:
        """Offset from origin along the normal (origin = point on plane)."""
        return jnp.einsum("...i,...i->...", self.normal, self.pose.translation())

    @staticmethod
    def from_point_and_normal(
        point: Float[Array, "*batch 3"], normal: Float[Array, "*batch 3"]
    ) -> HalfSpace:
        """Create a HalfSpace geometry from a point on the boundary and outward normal."""
        point, normal = jnp.array(point), jnp.array(normal)
        batch_axes = jnp.broadcast_shapes(point.shape[:-1], normal.shape[:-1])
        point = jnp.broadcast_to(point, batch_axes + (3,))
        normal = jnp.broadcast_to(normal, batch_axes + (3,))
        mat = make_frame(normal)
        pos = point
        pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_matrix(mat), pos
        )
        size = jnp.zeros(batch_axes + (1,), dtype=pos.dtype)
        return HalfSpace(pose=pose, size=size)

    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        """Visualize HalfSpace as a large thin box aligned with its boundary plane."""
        pose_i: jaxlie.SE3 = jax.tree.map(lambda x: x[index], self.pose)
        pos = onp.array(pose_i.translation())
        mat = onp.array(pose_i.rotation().as_matrix())
        # Visualize as a box representing the boundary plane
        plane_mesh = trimesh.creation.box(extents=[10, 10, 0.01])
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        plane_mesh.apply_transform(tf)
        return plane_mesh


@jdc.pytree_dataclass
class Sphere(CollGeom):
    """Sphere geometry."""

    @property
    def radius(self) -> Float[Array, "*batch"]:
        """Radius of the sphere."""
        return self.size[..., 0]

    @staticmethod
    def from_center_and_radius(
        center: Float[Array, "*batch 3"], radius: Float[Array, "*batch"]
    ) -> Sphere:
        """Create a Sphere geometry from a center point and radius."""
        center, radius = jnp.array(center), jnp.array(radius)
        batch_axes = jnp.broadcast_shapes(center.shape[:-1], radius.shape)
        center = jnp.broadcast_to(center, batch_axes + (3,))
        radius = jnp.broadcast_to(radius, batch_axes)
        pos = center
        # Create identity pose for sphere
        num_batch_elements = onp.prod(batch_axes).item() if batch_axes else 1
        quat_wxyz = jnp.stack(
            [jnp.array([1.0, 0.0, 0.0, 0.0], dtype=pos.dtype)] * num_batch_elements,
            axis=0,
        )
        quat_wxyz = quat_wxyz.reshape(batch_axes + (4,))
        wxyz_xyz = jnp.concatenate([quat_wxyz, pos], axis=-1)
        pose = jaxlie.SE3(wxyz_xyz)

        # Store radius in size[..., 0], shape_dim=1
        size = radius[..., None]
        return Sphere(pose=pose, size=size)

    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        pose_i: jaxlie.SE3 = jax.tree.map(lambda x: x[index], self.pose)
        pos = onp.array(pose_i.translation())
        radius_val = float(self.radius[index])
        sphere_mesh = trimesh.creation.icosphere(radius=radius_val, subdivisions=1)
        # Only apply translation for sphere
        tf = onp.eye(4)
        tf[:3, 3] = pos
        sphere_mesh.apply_transform(tf)
        return sphere_mesh


@jdc.pytree_dataclass
class Capsule(CollGeom):
    """Capsule geometry."""

    @property
    def radius(self) -> Float[Array, "*batch"]:
        """Radius of the capsule ends and cylinder."""
        return self.size[..., 0]

    @property
    def height(self) -> Float[Array, "*batch"]:
        """Height of the cylindrical segment."""
        return self.size[..., 1]

    @property
    def axis(self) -> Float[Array, "*batch 3"]:
        """Axis direction (Z-axis of rotation matrix)."""
        return self.pose.rotation().as_matrix()[..., :, 2]

    @staticmethod
    def from_radius_height(
        radius: Float[Array, "*batch"],
        height: Float[Array, "*batch"],  # Full height
        position: Float[Array, "*batch 3"] | None = None,
        wxyz: Float[Array, "*batch 4"] | None = None,
    ) -> Capsule:
        """Create Capsule geometry from radius and height."""
        if position is None:
            position = jnp.zeros((3,))
        if wxyz is None:
            wxyz = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity matrix.

        position = jnp.array(position)
        wxyz = jnp.array(wxyz)
        radius = jnp.array(radius)
        height = jnp.array(height)

        batch_axes = jnp.broadcast_shapes(
            position.shape[:-1], wxyz.shape[:-1], radius.shape, height.shape
        )
        pos = jnp.broadcast_to(position, batch_axes + (3,))
        wxyz = jnp.broadcast_to(wxyz, batch_axes + (4,))
        radius = jnp.broadcast_to(radius, batch_axes)
        height = jnp.broadcast_to(height, batch_axes)

        wxyz_xyz = jnp.concatenate([wxyz, pos], axis=-1)
        pose = jaxlie.SE3(wxyz_xyz)

        size = jnp.stack([radius, height], axis=-1)
        return Capsule(pose=pose, size=size)

    @staticmethod
    def from_trimesh(mesh: trimesh.Trimesh) -> Capsule:
        """
        Create Capsule geometry from minimum bounding cylinder of the mesh.
        """
        if mesh.is_empty:
            return Capsule(pose=jaxlie.SE3.identity(), size=jnp.zeros((2,)))
        results = trimesh.bounds.minimum_cylinder(mesh)
        radius = results["radius"]
        height = results["height"]
        tf_mat = results["transform"]
        tf = jaxlie.SE3.from_matrix(tf_mat)
        capsule = Capsule.from_radius_height(
            position=jnp.zeros((3,)),
            wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]),
            radius=radius,
            height=height,
        )
        capsule = capsule.transform(tf)
        return capsule

    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        pose_i: jaxlie.SE3 = jax.tree.map(lambda x: x[index], self.pose)
        pos = onp.array(pose_i.translation())
        mat = onp.array(pose_i.rotation().as_matrix())
        radius_val = float(self.radius[index])
        height_val = abs(float(self.height[index])) / 2

        # Create sphere and stretch it to match capsule shape.
        capsule_mesh = trimesh.creation.icosphere(radius=radius_val, subdivisions=1)
        capsule_mesh.vertices = onp.where(
            capsule_mesh.vertices[:, 2][..., None] > 0,
            capsule_mesh.vertices + onp.array([0.0, 0.0, height_val]),
            capsule_mesh.vertices - onp.array([0.0, 0.0, height_val]),
        )

        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        capsule_mesh.apply_transform(tf)
        return capsule_mesh

    def decompose_to_spheres(self, n_segments: int) -> Sphere:
        """
        Decompose the capsule into a series of spheres along its axis.
        Args: n_segments: Number of spheres.
        Returns: Sphere object shape (n_segments, *batch, ...).
        """
        batch_axes = self.get_batch_axes()
        radii = self.radius

        # Calculate local offsets for sphere centers along z-axis.
        segment_factors = jnp.linspace(-1.0, 1.0, n_segments)
        local_offsets_vec = jnp.array([0.0, 0.0, 1.0])[None, None, :] * (
            segment_factors[:, None, None] * self.height[None, ..., None] / 2
        )

        # Create base spheres (at origin, correct radius) and transform them.
        spheres = Sphere.from_center_and_radius(
            center=jnp.zeros((n_segments,) + batch_axes + (3,)),
            radius=jnp.broadcast_to(radii, (n_segments,) + batch_axes),
        )

        # Broadcast capsule pose and apply transforms.
        capsule_pose_broadcast = jaxlie.SE3(
            jnp.broadcast_to(
                self.pose.wxyz_xyz,
                (n_segments,) + self.pose.get_batch_axes() + (7,),
            )
        )
        spheres = spheres.transform(
            capsule_pose_broadcast @ jaxlie.SE3.from_translation(local_offsets_vec)
        )
        assert spheres.get_batch_axes() == (n_segments,) + batch_axes
        return spheres

    @staticmethod
    def from_sphere_pairs(sph_0: Sphere, sph_1: Sphere) -> Capsule:
        """
        Create a capsule connecting the centers of two spheres.
        Args: sph_0, sph_1: Input spheres.
        Returns: Capsule object with the same batch shape.
        """
        assert sph_0.get_batch_axes() == sph_1.get_batch_axes(), "Batch axes mismatch"

        pos0 = sph_0.pose.translation()
        pos1 = sph_1.pose.translation()
        vec = pos1 - pos0

        # Get height, safely handle zero-length case.
        x = pos1 - pos0
        is_zero = jnp.allclose(x, 0.0)
        x = jnp.where(is_zero, jnp.ones_like(x), x)
        n = jnp.linalg.norm(x + 1e-6, axis=-1, keepdims=True)
        height = jax.lax.select(is_zero, jnp.zeros_like(n), n).squeeze(-1)

        transform = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3.from_matrix(make_frame(vec)),
            translation=(pos0 + pos1) / 2.0,
        )

        capsule = Capsule.from_radius_height(
            position=transform.translation(),
            wxyz=transform.rotation().wxyz,
            radius=sph_0.radius,
            height=height,
        )

        assert capsule.get_batch_axes() == sph_0.get_batch_axes()
        return capsule


@jdc.pytree_dataclass
class BoundingBox(CollGeom):
    """Bounding box geometry."""

    @property
    def extents(self) -> Float[Array, "*batch 3"]:
        """Extents of the bounding box."""
        return self.size[..., :3]

    @property
    def center(self) -> Float[Array, "*batch 3"]:
        """Center of the bounding box."""
        return self.pose.translation()

    @staticmethod
    def from_center_and_extents(
        center: Float[Array, "*batch 3"], extents: Float[Array, "*batch 3"]
    ) -> BoundingBox:
        """Create a BoundingBox geometry from a center and extents."""
        center, extents = jnp.array(center), jnp.array(extents)
        batch_axes = jnp.broadcast_shapes(center.shape[:-1], extents.shape[:-1])
        center = jnp.broadcast_to(center, batch_axes + (3,))
        extents = jnp.broadcast_to(extents, batch_axes + (3,))
        return BoundingBox(pose=jaxlie.SE3.from_translation(center), size=extents)

    @staticmethod
    def from_trimesh(mesh: trimesh.Trimesh | list[trimesh.Trimesh]) -> BoundingBox:
        """Create a BoundingBox geometry from a trimesh object."""
        if isinstance(mesh, list):
            return cat_geoms([BoundingBox.from_trimesh(m) for m in mesh])

        mesh_min = mesh.bounds[0]
        mesh_max = mesh.bounds[1]
        extents = mesh_max - mesh_min
        center = (mesh_min + mesh_max) / 2.0
        pose = jaxlie.SE3.from_translation(center)
        return BoundingBox(pose=pose, size=extents)

    def _create_one_mesh(self, index: tuple) -> trimesh.Trimesh:
        pose_i: jaxlie.SE3 = jax.tree.map(lambda x: x[index], self.pose)
        pos = onp.array(pose_i.translation())
        extents_val = onp.array(self.extents[index])
        bbox_mesh = trimesh.creation.box(extents=extents_val)
        tf = onp.eye(4)
        tf[:3, 3] = pos
        bbox_mesh.apply_transform(tf)
        return bbox_mesh
