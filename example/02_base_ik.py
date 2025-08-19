import jax
import time
import viser
from soul.robots.cc_robot import CCRobot
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
    # Setup Environment
    robot = CCRobot.from_config("configs/robots/cc.json")
    robot_coll = RobotCollision.from_config("configs/robots/cc.json")

    # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot, robot_coll, root_node_name="/robot")
    robot_vis.create_robot_visualizations()

    # Setup GUI
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.3,
        position=(
            0.0,
            0.0,
            robot.config.length * robot.config.num_sections,
        ),
        wxyz=(1, 0, 0, 0),
    )
    server.scene.add_grid("/ground", width=6, height=6)

    # Setup IK Solver
    solver = IKSolver(
        robot, num_seeds_init=10, num_seeds_final=1, total_steps=64, init_steps=6
    )
    ik_solver = jax.jit(solver.solve_ik_best)

    while True:
        cfg = ik_solver(ik_target_handle.wxyz, ik_target_handle.position)
        pose = robot.forward_kinematics(cfg)
        robot_vis.update_pose(pose)


if __name__ == "__main__":
    viser_main()
