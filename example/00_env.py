from soul.envs.obs_env import ObstacleEnv
from soul.geom import Capsule, HalfSpace, Sphere, BoundingBox
import trimesh
import viser
import time
import numpy as np
import coacd


def test_convex_decomp() -> None:
    mesh = trimesh.load("assets/objects/warehouse_shelf.glb", force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(mesh, max_convex_hull=20)
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=6, height=6)
    for i, part in enumerate(parts):
        vertices, faces = part
        part_mesh = trimesh.Trimesh(vertices, faces)
        rot_90_x = trimesh.transformations.rotation_matrix(np.deg2rad(90), [1, 0, 0])
        part_mesh.apply_transform(rot_90_x)
        bbox = BoundingBox.from_trimesh(part_mesh)
        server.scene.add_mesh_trimesh(f"/obstacle_{i}/mesh", mesh=part_mesh)
        server.scene.add_mesh_trimesh(
            f"/obstacle_{i}/collision", mesh=bbox.to_trimesh()
        )

    while True:
        time.sleep(0.01)


def test_env() -> None:
    env = ObstacleEnv("configs/maps/obstacles.json")
    env.show()


if __name__ == "__main__":
    test_convex_decomp()
