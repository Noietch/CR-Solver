import jax
import time
import viser
import jaxlie
import numpy as np
from soul.robots.pcc_robot import PCCRobot
from soul.geom import HalfSpace, RobotCollision, Sphere
from soul.solver import ConstrainedMotionPlanner
from soul.visualization.visualizer_viser import ViserSoftRobot

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def get_linear_traj(
    start_position: np.ndarray,
    start_wxyz: np.ndarray,
    end_position: np.ndarray,
    end_wxyz: np.ndarray,
    timesteps: int,
) -> jaxlie.SE3:
    start_position = np.array(start_position)
    start_wxyz = np.array(start_wxyz)
    end_position = np.array(end_position)
    end_wxyz = np.array(end_wxyz)
    traj_positions = np.linspace(start_position, end_position, timesteps)
    traj_wxyz = np.linspace(start_wxyz, end_wxyz, timesteps)
    traj = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=traj_wxyz), translation=traj_positions
    )
    return traj


def main():
    pass


def viser_main():
    # Setup Environment
    robot = PCCRobot.from_config("configs/robots/pcc.json")
    robot_coll = RobotCollision.from_config("configs/robots/pcc.json")
    server = viser.ViserServer()
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.2])
    )
    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")
    start_handle = server.scene.add_transform_controls(
        "/start",
        scale=0.3,
        position=(1.0, 0.0, 2.5),
        wxyz=(1, 0, 0, 0),
    )
    end_handle = server.scene.add_transform_controls(
        "/end", scale=0.3, position=(0.0, -1.0, 2.5), wxyz=(1, 0, 0, 0)
    )
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.6, position=(0.4, 0.3, 0.4)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())
    server.scene.add_grid("/ground", width=6, height=6)
    plan_button = server.gui.add_button("Plan", disabled=False)
    replay_button = server.gui.add_button("Replay", disabled=False)
    # Set up trajopt parameters
    timesteps = 100
    traj_solver = ConstrainedMotionPlanner(robot, robot_coll, timesteps)
    traj_follow_jit = jax.jit(traj_solver.traj_follow)

    traj = None

    def plan_callback(args):
        print("Start planning....")
        global traj
        sphere_coll_world_current = sphere_coll.transform_from_pos_wxyz(
            position=np.array(sphere_handle.position),
            wxyz=np.array(sphere_handle.wxyz),
        )

        world_coll = [sphere_coll_world_current, plane_coll]

        reference_traj = get_linear_traj(
            start_handle.position,
            start_handle.wxyz,
            end_handle.position,
            end_handle.wxyz,
            timesteps,
        )
        cfg = traj_follow_jit(reference_traj, world_coll)
        traj = robot.forward_kinematics(cfg)
        print("Finish planning....")
        robot_vis.visualize_traj_collisions(robot, cfg)
        for i in range(timesteps):
            time.sleep(0.01)
            robot_vis.update_pose(traj[i])

    def replay_callback(args):
        global traj
        if traj is None:
            return
        for i in range(timesteps):
            time.sleep(0.01)
            robot_vis.update_pose(traj[i])

    plan_button.on_click(plan_callback)
    replay_button.on_click(replay_callback)

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    viser_main()
