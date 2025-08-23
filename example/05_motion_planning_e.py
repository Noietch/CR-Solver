import jax
import time
import viser
import numpy as np
from soul.robots.cc_robot import CCRobot
from soul.robots.tdcr_robot import TDCRRobot
from soul.geom import RobotCollision, WorldCollision, Sphere
from soul.solver import (
    TrajOptimizer,
    TrajOptimizerOptions,
)
from soul.visualization.visualizer_viser import (
    ViserSoftRobot,
    ViserWorld,
    ViserRenderer,
)


def viser_main(robot_type: str = "cc", default_method: str = "trajopt"):
    # Setup Robot Environment
    robot = CCRobot.from_config("configs/robots/cc.json")
    robot_coll = RobotCollision.from_config("configs/robots/cc.json")
    world_coll = WorldCollision.from_config(
        "configs/maps/ik_maps/obstacles_lattice.json"
    )

    # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot, robot_coll, root_node_name="/robot")
    robot_vis.create_robot_visualizations()
    obstacles_vis = ViserWorld(server, world_coll, enable_collision=False)
    obstacles_vis.create_mesh_visualizations()
    renderer = ViserRenderer(server, robot_vis, obstacles_vis)

    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.2])
    )
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.1, position=(-0.11366828, 0.67437919, 1.04626801)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())
    plan_button = server.gui.add_button("Plan", disabled=False)
    replay_button = server.gui.add_button("Replay", disabled=False)

    start_handle = server.scene.add_transform_controls(
        "/start",
        scale=0.3,
        position=(1, 0, 2),
        wxyz=(1, 0, 0, 0),
    )
    end_handle = server.scene.add_transform_controls(
        "/end", scale=0.3, position=(0, 1, 2), wxyz=(1, 0, 0, 0)
    )

    timesteps = 100
    options = TrajOptimizerOptions(
        collision_weight=0.0,
        smoothness_weight=50,
        pose_position_weight=200,
        pose_orientation_weight=100,
    )
    traj_solver = TrajOptimizer(robot, robot_coll, timesteps, options)
    traj_jit = jax.jit(traj_solver.optimize)
    start_end_interpolate_jit = jax.jit(traj_solver.start_end_interpolate)

    traj = None

    def plan_callback(args):
        sphere_coll_world_current = sphere_coll.transform_from_pos_wxyz(
            position=np.array(sphere_handle.position),
            wxyz=np.array(sphere_handle.wxyz),
        )
        world_coll_list = [sphere_coll_world_current]

        print("Start planning....")
        global traj
        cfg = start_end_interpolate_jit(
            start_handle.position,
            start_handle.wxyz,
            end_handle.position,
            end_handle.wxyz,
            world_coll_list,
        )
        cfg = traj_jit(
            cfg,
            world_coll_list,
            limit_weight=100.0,
            smoothness_weight=20.0,
            trajectory_length_weight=50,
            collision_weight=100,
            start_pose_weight=100,
            end_pose_weight=100,
        )
        traj = robot.forward_kinematics(cfg)
        print("Finish planning....")
        robot_vis.visualize_tip_traj(
            traj, color=np.array([0.0, 0.0, 1.0]), name="planned_traj"
        )
        for i in range(timesteps):
            time.sleep(0.01)
            robot_vis.update_pose(traj[i])

    def replay_callback(event):
        global traj
        if traj is None:
            return
        renderer.render_traj_image(event, traj, skip_frames=15, save_path=None)

    plan_button.on_click(plan_callback)
    replay_button.on_click(replay_callback)
    while True:
        time.sleep(1 / 60.0)


if __name__ == "__main__":
    viser_main(robot_type="tdcr", default_method="trajopt")
