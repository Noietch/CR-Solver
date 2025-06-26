from __future__ import annotations

import json
import trimesh
import jax_dataclasses as jdc
import jaxlie
import jax.numpy as jnp
from jaxtyping import Float, Array

from .geometry import BoundingBox, cat_geoms, HalfSpace, CollGeom
from .utils import load_mesh
import jaxlie


@jdc.pytree_dataclass
class WorldCollision:
    """World collision is a collection of obstacles bbox."""

    mesh: list[trimesh.Trimesh]
    obstacles: BoundingBox
    ground: HalfSpace

    @classmethod
    def from_config(cls, config: dict | str) -> WorldCollision:
        if isinstance(config, str):
            with open(config, "r") as f:
                config = json.load(f)
            
        obstacles = []
        meshes = []
        for obstacle in config.values():
            if obstacle["type"] == "bbox":
                bbox = BoundingBox.from_center_and_extents(
                    obstacle["position"], obstacle["extents"]
                )
                obstacles.append(bbox)
                meshes.append(bbox.to_trimesh())
            elif obstacle["type"] == "mesh":
                decompose_type = obstacle.get("decompose_type", None)
                decompose_params = obstacle.get("decompose_params", None)
                decomposed_mesh, original_mesh = load_mesh(
                    obstacle["path"],
                    scale=obstacle["scale"],
                    wxyz=obstacle["wxyz"],
                    position=obstacle["position"],
                    decompose_type=decompose_type,
                    decompose_params=decompose_params,
                )
                obstacles.append(BoundingBox.from_trimesh(decomposed_mesh))
                meshes.append(original_mesh)
            else:
                raise ValueError(f"Unknown obstacle type: {obstacle['type']}")
        if len(obstacles) == 1:
            obstacles = obstacles[0]
        obstacles = cat_geoms(obstacles)
        ground = HalfSpace.from_point_and_normal(
            jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
        )
        return cls(obstacles=obstacles, mesh=meshes, ground=ground)

    @property
    def collision_geoms(self) -> list[CollGeom]:
        return [self.obstacles, self.ground]


    def transform(self, position: Float[Array, "*batch 3"], wxyz: Float[Array, "*batch 4"]) -> WorldCollision:
        transform = jaxlie.SE3.from_rotation_and_translation(rotation=jaxlie.SO3(wxyz=jnp.array(wxyz)), translation=jnp.array(position))
        return WorldCollision(
            obstacles=self.obstacles.transform(transform),
            mesh=[m.copy().apply_transform(transform.as_matrix()) for m in self.mesh],
        )