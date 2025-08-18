import jax
import time
import viser
import numpy as np
import os
from soul.robots.cc_robot import CCRobot
from soul.robots.tdcr_robot import TDCRRobot
from soul.geom import RobotCollision, WorldCollision
from soul.solver import (
    TrajOptimizer,
    TrajOptimizerOptions,
    ParallelPRM,
    PRMOptions,
    OptimizedRRT,
    RRTOptions,
)
from soul.solver.motion_planner import resample_trajectory
from soul.visualization.visualizer_viser import (
    ViserSoftRobot,
    ViserWorld,
    ViserRenderer,
)

# Initialize JAX persistent compilation cache
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
from jax.experimental.compilation_cache import compilation_cache as cc

cc.set_cache_dir("/tmp/jax_cache")
DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def viser_main(robot_type: str = "cc", default_method: str = "trajopt"):
    # Setup Robot Environment
    if robot_type == "cc":
        robot = CCRobot.from_config("configs/robots/cc.json")
        robot_coll = RobotCollision.from_config("configs/robots/cc.json")
    elif robot_type == "tdcr":
        robot = TDCRRobot.from_config("configs/robots/cc_tdcr.json")
        robot_coll = RobotCollision.from_config("configs/robots/cc_tdcr.json")
    world_coll = WorldCollision.from_config("configs/maps/mp_scene/mp_demo.json")

    # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot, robot_coll, root_node_name="/robot")
    robot_vis.create_robot_visualizations()
    obstacles_vis = ViserWorld(server, world_coll, enable_collision=False)
    obstacles_vis.create_mesh_visualizations()
    renderer = ViserRenderer(server, robot_vis, obstacles_vis)

    # Setup GUI
    start_handle = server.scene.add_transform_controls(
        "/start",
        scale=0.3,
        position=(-0.3, -1.26, 2.51),
        wxyz=(1, 0, 0, 0),
    )
    end_handle = server.scene.add_transform_controls(
        "/end", scale=0.3, position=(-0.4, 1.45, 0.89), wxyz=(1, 0, 0, 0)
    )

    with server.gui.add_folder("Handles Tfs"):
        start_pos_text = server.gui.add_text(
            "Start Pos",
            initial_value=str(tuple(np.round(start_handle.position, 2))),
            disabled=True,
        )
        start_wxyz_text = server.gui.add_text(
            "Start wxyz",
            initial_value=str(tuple(np.round(start_handle.wxyz, 2))),
            disabled=True,
        )
        end_pos_text = server.gui.add_text(
            "End Pos",
            initial_value=str(tuple(np.round(end_handle.position, 2))),
            disabled=True,
        )
        end_wxyz_text = server.gui.add_text(
            "End wxyz",
            initial_value=str(tuple(np.round(end_handle.wxyz, 2))),
            disabled=True,
        )

    # Set up trajopt parameters
    timesteps = 100

    # Initialize trajectory optimizer options
    traj_options = TrajOptimizerOptions()

    with server.gui.add_folder("Planning Options"):
        method_dropdown = server.gui.add_dropdown(
            "Planning Method",
            options=["trajopt", "rrt", "prm"],
            initial_value=default_method,
            hint="Select the motion planning algorithm",
        )
        use_trajopt_after_planner = server.gui.add_checkbox(
            "Optimize after RRT/PRM",
            initial_value=True,
            hint="Apply trajectory optimization after RRT/PRM planning",
        )

    # Add cost weight controls
    with server.gui.add_folder("Cost Weights"):
        # Basic costs
        limit_weight_slider = server.gui.add_slider(
            "Limit Weight",
            min=0.0,
            max=100.0,
            step=1.0,
            initial_value=traj_options.limit_weight,
            hint="Weight for joint limit constraints",
        )
        smoothness_weight_slider = server.gui.add_slider(
            "Smoothness Weight",
            min=0.0,
            max=100.0,
            step=1.0,
            initial_value=traj_options.smoothness_weight,
            hint="Weight for trajectory smoothness",
        )
        traj_length_weight_slider = server.gui.add_slider(
            "Trajectory Length Weight",
            min=0.0,
            max=100.0,
            step=1.0,
            initial_value=traj_options.trajectory_length_weight,
            hint="Weight for minimizing trajectory length",
        )

        # Collision weight
        collision_weight_slider = server.gui.add_slider(
            "Collision Weight",
            min=0.0,
            max=400.0,
            step=5.0,
            initial_value=traj_options.collision_weight,
            hint="Weight for collision avoidance",
        )

        # Constraint weights
        start_pose_weight_slider = server.gui.add_slider(
            "Start Pose Weight",
            min=0.0,
            max=500.0,
            step=10.0,
            initial_value=traj_options.start_pose_weight,
            hint="Weight for start pose constraint",
        )
        end_pose_weight_slider = server.gui.add_slider(
            "End Pose Weight",
            min=0.0,
            max=500.0,
            step=10.0,
            initial_value=traj_options.end_pose_weight,
            hint="Weight for end pose constraint",
        )

        # TDCR-specific weights (only show for TDCR robot)
        if robot_type == "tdcr":
            tendon_vel_weight_slider = server.gui.add_slider(
                "Tendon Velocity Weight",
                min=0.0,
                max=50.0,
                step=1.0,
                initial_value=traj_options.tendon_vel_weight,
                hint="Weight for tendon velocity smoothness",
            )
            tendon_acc_weight_slider = server.gui.add_slider(
                "Tendon Acceleration Weight",
                min=0.0,
                max=50.0,
                step=1.0,
                initial_value=traj_options.tendon_acc_weight,
                hint="Weight for tendon acceleration smoothness",
            )
            dt_slider = server.gui.add_slider(
                "Time Step (dt)",
                min=0.01,
                max=1.0,
                step=0.01,
                initial_value=traj_options.dt,
                hint="Time step for time-based costs",
            )

        # Add reset button for weights
        reset_weights_button = server.gui.add_button("Reset Weights to Default")

        def reset_weights_callback(args):
            """Reset all weights to default values."""
            default_options = TrajOptimizerOptions()
            limit_weight_slider.value = default_options.limit_weight
            smoothness_weight_slider.value = default_options.smoothness_weight
            traj_length_weight_slider.value = default_options.trajectory_length_weight
            collision_weight_slider.value = default_options.collision_weight
            start_pose_weight_slider.value = default_options.start_pose_weight
            end_pose_weight_slider.value = default_options.end_pose_weight
            if robot_type == "tdcr":
                tendon_vel_weight_slider.value = default_options.tendon_vel_weight
                tendon_acc_weight_slider.value = default_options.tendon_acc_weight
                dt_slider.value = default_options.dt
            print("Weights reset to default values")

        reset_weights_button.on_click(reset_weights_callback)

    plan_button = server.gui.add_button("Plan", disabled=False)
    replay_button = server.gui.add_button("Replay", disabled=False)
    render_video_button = server.gui.add_button("Render Video", disabled=False)
    render_image_button = server.gui.add_button("Render Image", disabled=False)

    # init motion planners
    traj_solver = TrajOptimizer(robot, robot_coll, timesteps, options=traj_options)
    start_end_interpolate_jit = jax.jit(traj_solver.start_end_interpolate)
    start_end_ik_solver = traj_solver._ik_solver_best
    if robot_type == "cc":
        # JIT compile without making options static - they will be traced as dynamic values
        traj_opt = jax.jit(traj_solver.optimize)
    elif robot_type == "tdcr":
        # JIT compile without making options static - they will be traced as dynamic values
        traj_opt = jax.jit(traj_solver.optimize_tdcr)
    forward_kinematics = jax.jit(jax.vmap(robot._forward_kinematics))

    # Initialize RRT solver
    rrt_options = RRTOptions(
        batch_size=100,
        max_iterations=1000,
    )
    rrt_traj_solver = OptimizedRRT(robot, robot_coll, rrt_options)

    # Initialize PRM solver
    prm_options = PRMOptions(batch_size=2000, parallel_edge_checks=200)
    prm_traj_solver = ParallelPRM(robot, robot_coll, prm_options)
    if os.path.exists("roadmap_opt.pkl"):
        print("Loading existing roadmap...")
        prm_traj_solver.load_roadmap("roadmap_opt.pkl")
    else:
        print("Building new roadmap...")
        prm_traj_solver.build_roadmap(1000, world_coll.collision_geoms)
        prm_traj_solver.save_roadmap("roadmap_opt.pkl")

    traj = None
    print("Init done")

    def plan_callback(args):
        print("Start planning....")
        global traj

        # Update trajectory optimizer options from UI
        traj_options.limit_weight = limit_weight_slider.value
        traj_options.smoothness_weight = smoothness_weight_slider.value
        traj_options.trajectory_length_weight = traj_length_weight_slider.value
        traj_options.collision_weight = collision_weight_slider.value
        traj_options.start_pose_weight = start_pose_weight_slider.value
        traj_options.end_pose_weight = end_pose_weight_slider.value

        if robot_type == "tdcr":
            traj_options.tendon_vel_weight = tendon_vel_weight_slider.value
            traj_options.tendon_acc_weight = tendon_acc_weight_slider.value
            traj_options.dt = dt_slider.value

        # Get current options from GUI
        method = method_dropdown.value
        target_timesteps = timesteps

        print(f"Using method: {method}")
        if method == "trajopt":
            # Create optimizer with correct timesteps
            cfg = start_end_interpolate_jit(
                start_handle.position,
                start_handle.wxyz,
                end_handle.position,
                end_handle.wxyz,
                world_coll.collision_geoms,
            )
            # Optimize trajectory with current options
            if robot_type == "cc":
                cfg = traj_opt(
                    cfg,
                    world_coll.collision_geoms,
                    limit_weight=traj_options.limit_weight,
                    smoothness_weight=traj_options.smoothness_weight,
                    trajectory_length_weight=traj_options.trajectory_length_weight,
                    collision_weight=traj_options.collision_weight,
                    start_pose_weight=traj_options.start_pose_weight,
                    end_pose_weight=traj_options.end_pose_weight,
                )
            elif robot_type == "tdcr":
                cfg = traj_opt(
                    cfg,
                    world_coll.collision_geoms,
                    limit_weight=traj_options.limit_weight,
                    smoothness_weight=traj_options.smoothness_weight,
                    trajectory_length_weight=traj_options.trajectory_length_weight,
                    collision_weight=traj_options.collision_weight,
                    start_pose_weight=traj_options.start_pose_weight,
                    end_pose_weight=traj_options.end_pose_weight,
                    tendon_vel_weight=traj_options.tendon_vel_weight,
                    tendon_acc_weight=traj_options.tendon_acc_weight,
                    dt=traj_options.dt,
                )

        elif method == "prm":
            results = start_end_ik_solver(
                start_handle.wxyz,
                start_handle.position,
                end_handle.wxyz,
                end_handle.position,
                world_coll.collision_geoms,
            )
            cfg = prm_traj_solver.find_path(
                results[0],
                results[1],
                world_coll.collision_geoms,
            )
            if cfg is None:
                print("No path found")
                return

            # Resample PRM path to fixed timesteps
            print(f"PRM path length: {cfg.theta.shape[0]} timesteps")
            if use_trajopt_after_planner.value:
                cfg = resample_trajectory(cfg, target_timesteps)
                print(f"Resampled to: {cfg.theta.shape[0]} timesteps")
                if robot_type == "cc":
                    cfg = traj_opt(
                        cfg,
                        world_coll.collision_geoms,
                        limit_weight=traj_options.limit_weight,
                        smoothness_weight=traj_options.smoothness_weight,
                        trajectory_length_weight=traj_options.trajectory_length_weight,
                        collision_weight=traj_options.collision_weight,
                        start_pose_weight=traj_options.start_pose_weight,
                        end_pose_weight=traj_options.end_pose_weight,
                    )
                elif robot_type == "tdcr":
                    cfg = traj_opt(
                        cfg,
                        world_coll.collision_geoms,
                        limit_weight=traj_options.limit_weight,
                        smoothness_weight=traj_options.smoothness_weight,
                        trajectory_length_weight=traj_options.trajectory_length_weight,
                        collision_weight=traj_options.collision_weight,
                        start_pose_weight=traj_options.start_pose_weight,
                        end_pose_weight=traj_options.end_pose_weight,
                        tendon_vel_weight=traj_options.tendon_vel_weight,
                        tendon_acc_weight=traj_options.tendon_acc_weight,
                        dt=traj_options.dt,
                    )

        elif method == "rrt":
            results = start_end_ik_solver(
                start_handle.wxyz,
                start_handle.position,
                end_handle.wxyz,
                end_handle.position,
                world_coll.collision_geoms,
            )
            cfg = rrt_traj_solver.find_path(
                results[0],
                results[1],
                world_coll.collision_geoms,
            )
            if cfg is None:
                print("No path found")
                return

            # Resample RRT path to fixed timesteps
            print(f"RRT path length: {cfg.theta.shape[0]} timesteps")
            if use_trajopt_after_planner.value:
                cfg = resample_trajectory(cfg, target_timesteps)
                print(f"Resampled to: {cfg.theta.shape[0]} timesteps")
                if robot_type == "cc":
                    cfg = traj_opt(
                        cfg,
                        world_coll.collision_geoms,
                        limit_weight=traj_options.limit_weight,
                        smoothness_weight=traj_options.smoothness_weight,
                        trajectory_length_weight=traj_options.trajectory_length_weight,
                        collision_weight=traj_options.collision_weight,
                        start_pose_weight=traj_options.start_pose_weight,
                        end_pose_weight=traj_options.end_pose_weight,
                    )
                elif robot_type == "tdcr":
                    cfg = traj_opt(
                        cfg,
                        world_coll.collision_geoms,
                        limit_weight=traj_options.limit_weight,
                        smoothness_weight=traj_options.smoothness_weight,
                        trajectory_length_weight=traj_options.trajectory_length_weight,
                        collision_weight=traj_options.collision_weight,
                        start_pose_weight=traj_options.start_pose_weight,
                        end_pose_weight=traj_options.end_pose_weight,
                        tendon_vel_weight=traj_options.tendon_vel_weight,
                        tendon_acc_weight=traj_options.tendon_acc_weight,
                        dt=traj_options.dt,
                    )

        traj = forward_kinematics(cfg)
        traj = jax.block_until_ready(traj)
        print("Finish planning....")
        for i in range(len(traj)):
            time.sleep(1 / 60.0)
            robot_vis.update_pose(traj[i])

    def replay_callback(args):
        print("Start replaying....")
        global traj
        if traj is None:
            return
        for i in range(len(traj)):
            robot_vis.update_pose(traj[i])
        print("Finish replaying....")

    def render_video_callback(event: viser.GuiEvent):
        global traj
        if traj is None:
            return
        renderer.render_traj_video(event, traj, save_path="trajectory_video.mp4")

    def render_image_callback(event: viser.GuiEvent):
        global traj
        if traj is None:
            return
        renderer.render_traj_image(event, traj, save_path="trajectory_image.png")

    def on_handle_update(handle: viser.TransformControlsHandle):
        """Update GUI when handles are moved."""
        start_pos_text.value = str(np.round(start_handle.position, 2))
        start_wxyz_text.value = str(np.round(start_handle.wxyz, 2))
        end_pos_text.value = str(np.round(end_handle.position, 2))
        end_wxyz_text.value = str(np.round(end_handle.wxyz, 2))

    plan_button.on_click(plan_callback)
    replay_button.on_click(replay_callback)
    render_video_button.on_click(render_video_callback)
    render_image_button.on_click(render_image_callback)
    start_handle.on_update(on_handle_update)
    end_handle.on_update(on_handle_update)
    on_handle_update(start_handle)

    plan_callback(None)

    while True:
        time.sleep(1 / 60.0)


if __name__ == "__main__":
    viser_main(robot_type="tdcr", default_method="trajopt")
