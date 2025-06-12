"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time
import viser
import numpy as np
from soul.robots.pcc_robot import PCCRobot
from soul.solver import solve_ik
from soul.collision import HalfSpace, RobotCollision, Sphere
from soul.visualization.visualizer_viser import ViserSoftRobot
from soul.visualization.visualizer_plot import visualize_pcc_model_3d
from soul.envs.obs_env import ObstacleEnv
from soul.solver.traj_follow import solve_ee_traj_follow, solve_ee_traj_follow_dp



DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)

def circle_traj(traj_length:int):
    position_list, wxyz_list = [], []
    for i in range(traj_length):
        position = np.array([np.cos(i/traj_length*2*np.pi), np.sin(i/traj_length*2*np.pi), 2.5])
        wxyz = np.array([1, 0, 0, 0])
        position_list.append(position)
        wxyz_list.append(wxyz)
    return np.stack(position_list), np.stack(wxyz_list)


def square_traj(traj_length:int):
    position_list, wxyz_list = [], []
    
    # Define square parameters
    side_length = 1.0  # Size of the square
    center = np.array([0, 0, 2.5])  # Center of the square
    
    # Divide trajectory into 4 sides
    points_per_side = traj_length // 4
    
    for i in range(traj_length):
        side = i // points_per_side
        t = (i % points_per_side) / points_per_side if points_per_side > 0 else 0
        
        if side == 0:  # Bottom side: left to right
            position = center + np.array([-side_length/2 + t*side_length, -side_length/2, 0])
        elif side == 1:  # Right side: bottom to top
            position = center + np.array([side_length/2, -side_length/2 + t*side_length, 0])
        elif side == 2:  # Top side: right to left
            position = center + np.array([side_length/2 - t*side_length, side_length/2, 0])
        else:  # Left side: top to bottom
            position = center + np.array([-side_length/2, side_length/2 - t*side_length, 0])
        
        wxyz = np.array([1, 0, 0, 0])
        position_list.append(position)
        wxyz_list.append(wxyz)
    
    return np.stack(position_list), np.stack(wxyz_list)


def line_traj(traj_length:int):
    position_list, wxyz_list = [], []
    line_length = 1.0
    for i in range(traj_length):
        position = np.array([i/traj_length*line_length, 0, 2.5])
        wxyz = np.array([1, 0, 0, 0])
        position_list.append(position)
        wxyz_list.append(wxyz)
    return np.stack(position_list), np.stack(wxyz_list)

def main():
    robot_config = "configs/robots/pcc_2d.json"
    map_config = "configs/maps/obstacles.json"
    
    robot = PCCRobot.from_config(robot_config)
    num_points = robot.config.num_points_per_section
    robot_coll = RobotCollision.from_config(
        robot_config, self_collision_sampling_rate=1
    )
    ee_traj = line_traj(10)
    solution = solve_ee_traj_follow_dp(robot, ee_traj[0], ee_traj[1])
    fk_result = robot.forward_kinematics(solution)
    visualize_pcc_model_3d(fk_result, target_position=ee_traj[0], save_path="visualization/ee_traj_following.png", num_points=num_points)


def viser_main():
    robot_config = "configs/robots/pcc_2d.json"
    map_config = "configs/maps/obstacles.json"

    robot = PCCRobot.from_config(robot_config)
    robot_coll = RobotCollision.from_config(
        robot_config, self_collision_sampling_rate=1
    )
    server = viser.ViserServer(port=8081)
    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")

    plan_button = server.gui.add_button("Plan", disabled=False)
    replay_button = server.gui.add_button("Replay", disabled=False)
    timesteps = 100
    dt = 0.01
    fk_result = None

    def plan_callback(args):
        global fk_result
        ee_traj = circle_traj(timesteps)
        solution = solve_ee_traj_follow(robot, ee_traj[0], ee_traj[1])
        fk_result = robot.forward_kinematics(solution)
        for i in range(timesteps):
            time.sleep(dt)
            robot_vis.update_pose(fk_result[i])

    def replay_callback(args):
        global fk_result
        if fk_result is None:
            return
        for i in range(timesteps):
            time.sleep(dt)
            robot_vis.update_pose(fk_result[i])

    plan_button.on_click(plan_callback)
    replay_button.on_click(replay_callback)

    while True:
        time.sleep(0.01)

if __name__ == "__main__":
    main()
