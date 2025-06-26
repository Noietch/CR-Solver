import jax
import viser
from soul.robots.pcc_robot import PCCRobot
from soul.solver import IKSolver
from soul.geom import RobotCollision, WorldCollision
from soul.visualization.visualizer_viser import ViserSoftRobot, ViserWorld

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def viser_main():
    # Setup Environment
    robot = PCCRobot.from_config("configs/robots/pcc_mobile.json")
    robot_coll = RobotCollision.from_config("configs/robots/pcc_mobile.json")
    world_coll = WorldCollision.from_config("configs/maps/obstacles_01.json")

    # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")
    robot_vis.create_sphere_visualizations()
    obstacles_vis = ViserWorld(server, world_coll)
    obstacles_vis.create_mesh_visualizations()

    # Setup GUI
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.5,
        position=(0.0, 0.0, robot.config.length * robot.config.num_sections),
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
