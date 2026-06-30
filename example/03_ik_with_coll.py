import jax
import numpy as np
import viser

from soul.geom import HalfSpace, RobotCollision, Sphere
from soul.robots.cc_robot import CCRobot
from soul.solver import IKSolver
from soul.visualization.visualizer_viser import ViserSoftRobot

DISABLE_JIT = False

if DISABLE_JIT:
    import os

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def main():
    robot = CCRobot.from_config("configs/robots/cc.json")
    robot_coll = RobotCollision.from_config(
        "configs/robots/cc.json", self_collision_sampling_rate=1
    )
    server = viser.ViserServer()
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.2])
    )
    solver = IKSolver(
        robot,
        num_seeds_init=10,
        num_seeds_final=1,
        total_steps=64,
        init_steps=6,
        coll=robot_coll,
    )
    ik_solver = jax.jit(solver.solve_ik_best_with_coll)

    robot_vis = ViserSoftRobot(
        server, robot, robot_coll, root_node_name="/robot"
    )
    robot_vis.create_robot_visualizations()
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.8,
        position=(0.0, 0.0, robot.config.length * robot.config.num_sections),
        wxyz=(1, 0, 0, 0),
    )
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.1, position=(0.8, 0.03, 1.126)
    )
    server.scene.add_mesh_trimesh(
        "/obstacle/mesh", mesh=sphere_coll.to_trimesh()
    )
    server.scene.add_grid("/ground", width=6, height=6)
    server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    target_handle = server.gui.add_vector3(
        "Target", ik_target_handle.position, disabled=True
    )

    while True:
        sphere_coll_world_current = sphere_coll.transform_from_pos_wxyz(
            position=np.array(sphere_handle.position),
            wxyz=np.array(sphere_handle.wxyz),
        )
        world_coll_list = [plane_coll, sphere_coll_world_current]
        cfg = ik_solver(
            ik_target_handle.wxyz, ik_target_handle.position, world_coll_list
        )
        pose = robot.forward_kinematics(cfg)
        robot_vis.update_pose(pose)
        target_handle.value = ik_target_handle.position


if __name__ == "__main__":
    main()
