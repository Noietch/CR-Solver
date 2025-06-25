import jax
import time
import viser
from soul.robots.pcc_robot import PCCRobot
from soul.solver import IKSolver
from soul.geom import RobotCollision
from soul.visualization.visualizer_viser import ViserSoftRobot

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def viser_main():
    robot = PCCRobot.from_config("configs/robots/pcc.json")
    robot_coll = RobotCollision.from_config("configs/robots/pcc.json")
    solver = IKSolver(
        robot, num_seeds_init=10, num_seeds_final=1, total_steps=64, init_steps=6
    )
    ik_solver = jax.jit(solver.solve_ik_best)
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target",
        scale=1,
        position=(0.0, 0.0, robot.config.length * robot.config.num_sections),
        wxyz=(0, 0, 1, 0),
    )
    server.scene.add_grid("/ground", width=6, height=6)
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        start_time = time.time()
        cfg = ik_solver(ik_target_handle.wxyz, ik_target_handle.position)
        pose = robot.forward_kinematics(cfg)
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
        robot_vis.update_pose(pose)


if __name__ == "__main__":
    viser_main()
