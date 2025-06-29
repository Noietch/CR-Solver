import jax
import jaxlie
import time
import viser
import numpy as np
from soul.robots.pcc_robot import PCCRobot
from soul.geom import RobotCollision, WorldCollision
from soul.solver import ConstrainedMotionPlanner, IKSolver
from soul.visualization.visualizer_viser import ViserSoftRobot, ViserWorld

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


def get_sine_traj(
    start_position: np.ndarray,
    start_wxyz: np.ndarray,
    end_position: np.ndarray,
    end_wxyz: np.ndarray,
    timesteps: int,
):
    start_position = np.array(start_position)
    start_wxyz = np.array(start_wxyz)
    end_position = np.array(end_position)
    end_wxyz = np.array(end_wxyz)

    # Linear interpolation for y and z coordinates
    y_coords = np.linspace(start_position[1], end_position[1], timesteps)
    z_coords = np.linspace(start_position[2], end_position[2], timesteps)

    # Linear interpolation for x base values
    x_start = start_position[0]
    x_end = end_position[0]
    x_base = np.linspace(x_start, x_end, timesteps)

    # Add sinusoidal variation to x-axis
    # Create a sine wave that completes one full cycle along the z-axis trajectory
    t = np.linspace(0, 2 * np.pi, timesteps)
    x_amplitude = 0.5  # Amplitude of the sine wave
    x_sine = x_base + x_amplitude * np.sin(t)

    # Combine sinusoidal x, linear y, and linear z coordinates
    traj_positions = np.column_stack([x_sine, y_coords, z_coords])

    # Linear interpolation for rotation
    traj_wxyz = np.linspace(start_wxyz, end_wxyz, timesteps)

    # Create SE3 trajectory
    traj = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=traj_wxyz), translation=traj_positions
    )
    return traj


def viser_main():
    # Setup Environment
    robot = PCCRobot.from_config("configs/robots/pcc_mobile_z.json")
    robot_coll = RobotCollision.from_config("configs/robots/pcc_mobile_z.json")
    # world_coll = WorldCollision.from_config("configs/maps/obstacles_01.json")

    # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")
    robot_vis.create_sphere_visualizations()
    # obstacles_vis = ViserWorld(server, world_coll)
    # obstacles_vis.create_mesh_visualizations()

    # Setup GUI
    start_handle = server.scene.add_transform_controls(
        "/start",
        scale=0.3,
        position=(0.0, 0.0, 3.0),
        wxyz=(1, 0, 0, 0),
    )
    end_handle = server.scene.add_transform_controls(
        "/end", scale=0.3, position=(0.0, 0.0, 5.0), wxyz=(1, 0, 0, 0)
    )
    plan_button = server.gui.add_button("Plan", disabled=False)
    replay_button = server.gui.add_button("Replay", disabled=False)

    # Set up reference trajectory
    timesteps = 30

    def update_reference_traj(args):
        reference_traj = get_sine_traj(
            start_handle.position,
            start_handle.wxyz,
            end_handle.position,
            end_handle.wxyz,
            timesteps,
        )
        robot_vis.visualize_traj(
            reference_traj, color=np.array([1.0, 0.0, 0.0]), name="reference_traj"
        )
        return reference_traj

    update_reference_traj(None)

    # Set up trajopt parameters
    solver = IKSolver(
        robot, num_seeds_init=10, num_seeds_final=1, total_steps=64, init_steps=6, coll=robot_coll
    )
    ik_solver_jit = jax.jit(solver.solve_ik_best_with_coll_shape)
    traj_solver = ConstrainedMotionPlanner(robot, robot_coll, timesteps)
    traj_follow_jit = jax.jit(traj_solver.tip_traj_follow)

    traj = None

    def plan_callback(args):
        print("Start planning....")
        global traj

        # Update reference trajectory with current handle positions
        current_reference_traj = update_reference_traj(args)

        cfg = ik_solver_jit(
            current_reference_traj.as_matrix(),
            [],
            # world_coll.collision_geoms,
        )
        # cfg = traj_follow_jit(
        #     current_reference_traj,
        #     world_coll.collision_geoms,
        # )
        print(cfg)
        pose = robot.forward_kinematics(cfg)
        print("Finish planning....")
        robot_vis.update_pose(pose)
        # robot_vis.visualize_traj_collisions(robot, cfg)
        # robot_vis.visualize_traj(
        #     traj, color=np.array([0.0, 0.0, 1.0]), name="planned_traj"
        # )
        # for i in range(timesteps):
        #     time.sleep(0.01)
        #     robot_vis.update_pose(traj[i])

    def replay_callback(args):
        global traj
        if traj is None:
            return
        for i in range(timesteps):
            time.sleep(0.01)
            robot_vis.update_pose(traj[i])

    plan_button.on_click(plan_callback)
    replay_button.on_click(replay_callback)
    start_handle.on_update(update_reference_traj)
    end_handle.on_update(update_reference_traj)

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    viser_main()
