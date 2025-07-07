import jax
import viser
import time
import numpy as np
import jax.numpy as jnp
from jax import Array

from soul.robots import PCCRobot, PCCState
from soul.geom import RobotCollision, Sphere
from soul.visualization.visualizer_viser import ViserSoftRobot
from soul.solver.tendon_fk_solver import TendonFKSolver

# jax computation precision is different on cpu and gpu when using gpu
# set to highest to avoid numerical issues
jax.config.update("jax_default_matmul_precision", "highest")

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def tendon_similarity_cost(
    robot: PCCRobot,
    state: PCCState,
    tendon_target: Array,
    weight: Array | float,
) -> Array:
    jax.debug.print("state: {}", state)
    tendon_lengths = robot.compute_tendon_lengths(state)
    jax.debug.print("tendon_lengths: {}", tendon_lengths)
    jax.debug.print("tendon_target: {}", tendon_target)
    residual = tendon_lengths[0] - tendon_target[0]
    return (residual * weight).flatten()


def main():
    # load robot
    robot = PCCRobot.from_config("configs/robots/pcc.json")
    robot_coll = RobotCollision.from_config("configs/robots/pcc.json")
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.2])
    )
    sphere_coll_0 = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.2])
    )

    # load viser server
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")
    robot_vis.create_sphere_visualizations()
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.3, position=(0.8, 0, 0.8)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())
    sphere_handle_0 = server.scene.add_transform_controls(
        "/obstacle_0", scale=0.3, position=(0.8, 0, 0.8)
    )
    server.scene.add_mesh_trimesh("/obstacle_0/mesh", mesh=sphere_coll_0.to_trimesh())
    server.scene.add_grid("/ground", width=6, height=6)

    # load tendon fk solver
    tendon_fk_solver = TendonFKSolver(robot, robot_coll, total_steps=100)
    jit_tendon_fk_solver = jax.jit(tendon_fk_solver.pcc_fk)
    start_state = PCCState(
        base_position=jnp.zeros(3),
        kappa_x=jnp.zeros((robot.config.num_sections,)),
        kappa_y=jnp.zeros((robot.config.num_sections,)),
        epsilon=jnp.zeros((robot.config.num_sections,)),
    )

    # setup ui
    total_length = float(robot.config.length * robot.config.num_sections)
    slider = server.gui.add_slider(
        f"Tendon",
        min=0.0,
        max=total_length * 1.5,
        step=0.01,
        initial_value=total_length,
    )

    while True:
        sphere_coll_world_current = sphere_coll.transform_from_pos_wxyz(
            position=np.array(sphere_handle.position),
            wxyz=np.array(sphere_handle.wxyz),
        )
        sphere_coll_world_current_0 = sphere_coll_0.transform_from_pos_wxyz(
            position=np.array(sphere_handle_0.position),
            wxyz=np.array(sphere_handle_0.wxyz),
        )
        world_coll_list = [sphere_coll_world_current, sphere_coll_world_current_0]
        cfg = jit_tendon_fk_solver(
            start_state, world_coll_list, jnp.array([slider.value])
        )
        cost = tendon_similarity_cost(robot, cfg, jnp.array([slider.value]), 1.0)
        print(cost)
        pose = robot.forward_kinematics(cfg)
        robot_vis.update_pose(pose)
        time.sleep(0.01)


if __name__ == "__main__":
    main()
