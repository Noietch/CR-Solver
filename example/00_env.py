import time

import coacd
import numpy as np
import trimesh
import viser
from trimesh.voxel import creation

from cr_solver.geom import BoundingBox


def test_convex_decomp() -> None:
    mesh = trimesh.load("assets/objects/warehouse_shelf.glb", force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(mesh, max_convex_hull=20)
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=6, height=6)
    for i, part in enumerate(parts):
        vertices, faces = part
        part_mesh = trimesh.Trimesh(vertices, faces)
        rot_90_x = trimesh.transformations.rotation_matrix(
            np.deg2rad(90), [1, 0, 0]
        )
        part_mesh.apply_transform(rot_90_x)
        bbox = BoundingBox.from_trimesh(part_mesh)
        server.scene.add_mesh_trimesh(f"/obstacle_{i}/mesh", mesh=part_mesh)
        server.scene.add_mesh_trimesh(
            f"/obstacle_{i}/collision", mesh=bbox.to_trimesh()
        )

    while True:
        time.sleep(0.01)


def test_trimesh_to_voxel() -> None:
    mesh = trimesh.load("assets/objects/warehouse_shelf.glb", force="mesh")
    voxel_grid = creation.voxelize(mesh, pitch=0.4)
    voxel_cubes = []
    for center_point in voxel_grid.fill().points:
        single_cube = trimesh.creation.box()
        # Assign a random color to the cube for visualization
        color = np.random.rand(4)  # RGBA
        single_cube.visual.face_colors = (color * 255).astype(np.uint8)
        single_cube.apply_scale(voxel_grid.pitch)
        single_cube.apply_translation(center_point)
        voxel_cubes.append(single_cube)
    print("number of voxel_cubes: ", len(voxel_cubes))
    combined_mesh = trimesh.util.concatenate(voxel_cubes)
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=6, height=6)
    server.scene.add_mesh_trimesh("/obstacle", mesh=combined_mesh)
    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    # test_convex_decomp()
    test_trimesh_to_voxel()
