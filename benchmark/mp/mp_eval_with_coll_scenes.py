import jax
from jaxtyping import Array
import numpy as np
import jax.numpy as jnp
import jaxlie
import time
import json
import os
from typing import Callable, Sequence

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

from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.robots.tdcr_robot import TDCRRobot
from soul.solver import (
    TrajOptimizer,
    TrajOptimizerOptions,
    ParallelPRM,
    PRMOptions,
    OptimizedRRT,
    RRTOptions,
)
from soul.solver.motion_planner import resample_trajectory
from soul.geom import (
    RobotCollision,
    WorldCollision,
    CollGeom,
)
from benchmark.mp.mp_analyze import save_results
from benchmark.mp.utils import (
    mp_metric_with_coll,
    sample_collision_free_start_end_states,
    is_trajectory_in_collision,
    summarize_results,
)

jax.config.update("jax_default_matmul_precision", "highest")

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def _traj_optimize(
    robot_type: str,
    traj_opt: Callable,
    traj_options: TrajOptimizerOptions,
    world_coll: Sequence[CollGeom],
    cfg: Array,
) -> ConstantCurvatureState:
    """Optimizes a trajectory using the TrajOpt framework."""
    if robot_type == "cc":
        cfg = traj_opt(
            cfg,
            world_coll,
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
            world_coll,
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
    return cfg


def test_time_of_prm_opt(
    robot_type: str,
    solver: tuple[Callable, ParallelPRM],
    traj_opt_solver: tuple,  # (traj_opt_jit, traj_options)
    target_timesteps: int,
    start_position: Array,
    start_wxyz: Array,
    target_position: Array,
    target_wxyz: Array,
    world_geom: Sequence[CollGeom],
) -> tuple[float, float, float]:
    ik_solver, prm_solver = solver
    # Test ik time
    ik_start = time.time()
    cfg_pair = ik_solver(
        start_wxyz, start_position, target_wxyz, target_position, world_geom
    )
    jax.block_until_ready(cfg_pair)
    ik_end = time.time()

    # Test prm time
    cfg = prm_solver.find_path(cfg_pair[0], cfg_pair[1], world_geom)
    jax.block_until_ready(cfg)
    prm_end = time.time()
    ik_time = ik_end - ik_start
    prm_time = prm_end - ik_end
    if cfg is None:
        opt_time = float("nan")
        print("No path found")
        return ik_time, prm_time, opt_time
    traj_opt, traj_options = traj_opt_solver

    # Test optimize time
    opt_start = time.time()
    path_cfg = resample_trajectory(cfg, target_timesteps)
    path_cfg = _traj_optimize(robot_type, traj_opt, traj_options, world_geom, path_cfg)
    jax.block_until_ready(path_cfg)
    opt_time = time.time() - opt_start
    print(f"Finish optimizing PRM path with TrajOpt...")
    print(
        f"IK Time: {ik_time:.4f}s | PRM Time: {prm_time:.4f}s | Opt Time: {opt_time:.4f}s"
    )
    return ik_time, prm_time, opt_time


def _solve_with_trajopt(
    solver: tuple[Callable, Callable, TrajOptimizerOptions, str],
    start_position: Array,
    start_wxyz: Array,
    target_position: Array,
    target_wxyz: Array,
    world_geom: Sequence[CollGeom],
) -> ConstantCurvatureState:
    """Specific solving logic for TrajOpt."""
    start_end_interpolate_jit, traj_opt_jit, traj_options, robot_type = solver
    cfg = start_end_interpolate_jit(
        start_position, start_wxyz, target_position, target_wxyz, world_geom
    )

    path_cfg = _traj_optimize(robot_type, traj_opt_jit, traj_options, world_geom, cfg)
    return path_cfg


def _solve_with_prm(
    solver: tuple[Callable, ParallelPRM],
    start_position: Array,
    start_wxyz: Array,
    target_position: Array,
    target_wxyz: Array,
    world_geom: Sequence[CollGeom],
) -> ConstantCurvatureState:
    """Specific solving logic for PRM."""
    ik_solver, prm_solver = solver
    cfg_pair = ik_solver(
        start_wxyz, start_position, target_wxyz, target_position, world_geom
    )
    path_cfg = prm_solver.find_path(cfg_pair[0], cfg_pair[1], world_geom)
    return path_cfg


def _solve_with_rrt(
    solver: tuple[Callable, OptimizedRRT],
    start_position: Array,
    start_wxyz: Array,
    target_position: Array,
    target_wxyz: Array,
    world_geom: Sequence[CollGeom],
) -> ConstantCurvatureState:
    """Specific solving logic for RRT."""
    ik_solver, rrt_solver = solver
    cfg_pair = ik_solver(
        start_wxyz, start_position, target_wxyz, target_position, world_geom
    )
    path_cfg = rrt_solver.find_path(cfg_pair[0], cfg_pair[1], world_geom)
    return path_cfg


def _eval_planner(
    planner_name: str,
    robot_type: str,
    solver,
    solve_fn: Callable,
    target_timesteps: int,
    whether_use_trajopt_after_planner: bool,
    traj_opt_solver: tuple,  # (traj_opt_jit, traj_options)
    robot: CCRobot,
    batched_fk: Callable[[ConstantCurvatureState], Array],
    robot_coll: RobotCollision,
    world_geom: Sequence[CollGeom],
    start_states: ConstantCurvatureState,
    end_states: ConstantCurvatureState,
) -> tuple[dict, dict]:
    """Generic evaluation function for a single motion planning problem."""
    method_name_upper = planner_name.upper()

    # Get start and end poses from the states via FK
    start_transforms = batched_fk(start_states)
    start_tip_transform = jaxlie.SE3.from_matrix(start_transforms[0, -1, ...])
    start_wxyz = start_tip_transform.rotation().wxyz
    start_position = start_tip_transform.translation()

    target_transforms = batched_fk(end_states)
    target_tip_transform = jaxlie.SE3.from_matrix(target_transforms[0, -1, ...])
    target_wxyz = target_tip_transform.rotation().wxyz
    target_position = target_tip_transform.translation()

    # Call the specific solver function and time it
    start_time = time.time()
    path_cfg = solve_fn(
        solver, start_position, start_wxyz, target_position, target_wxyz, world_geom
    )
    jax.block_until_ready(path_cfg)
    total_time = time.time() - start_time

    # Post-process with TrajOpt if enabled
    if (
        (planner_name == "prm" or planner_name == "rrt")
        and whether_use_trajopt_after_planner
        and path_cfg is not None
        and path_cfg.theta.shape[0] > 0
    ):
        total_time = 0.0
        path_cfg = resample_trajectory(path_cfg, target_timesteps)
        traj_opt, traj_options = traj_opt_solver

        path_cfg = _traj_optimize(
            robot_type, traj_opt, traj_options, world_geom, path_cfg
        )
        jax.block_until_ready(path_cfg)
        total_time = time.time() - start_time
        print(f"Finish optimizing {planner_name.upper()} path with TrajOpt...")

    # Process results
    is_valid = path_cfg is not None and path_cfg.theta.shape[0] > 0

    # This dictionary will hold the data to be saved later.
    data_to_save = {}

    if is_valid:
        solution_states = jax.tree_util.tree_map(lambda x: x[-1:], path_cfg)
        all_paths = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), path_cfg
        )
        paths_are_valid = jnp.array([True])

        fk_result = batched_fk(path_cfg)
        planned_tip_traj = jaxlie.SE3.from_matrix(fk_result[:, -1, ...])

        # Prepare data for saving
        data_to_save = {
            "start_states_theta": np.asarray(start_states.theta),
            "start_states_phi": np.asarray(start_states.phi),
            "end_states_theta": np.asarray(end_states.theta),
            "end_states_phi": np.asarray(end_states.phi),
            "start_position": np.asarray(start_position),
            "start_wxyz": np.asarray(start_wxyz),
            "target_position": np.asarray(target_position),
            "target_wxyz": np.asarray(target_wxyz),
            "fk_result": np.asarray(fk_result),
            "solution_states_theta": np.asarray(solution_states.theta),
            "solution_states_phi": np.asarray(solution_states.phi),
            "planned_paths": all_paths.save_dict(),
            "planned_tip_traj": np.asarray(planned_tip_traj.as_matrix()),
        }

    else:
        print(f"[Warning] No solution found by {method_name_upper}")
        # Return summary and empty data dictionary
        summary = {
            "method": method_name_upper,
            "eval num": 1,
            "actual eval num": 1,
            "num sections": robot.config.num_sections,
            "kinematic_reachability_rate": 0.0,
            "position error": np.nan,
            "rotation error": np.nan,
            "success rate": 0.0,
            "total time": total_time,
            "failure_stats": {
                "total_samples": 1,
                "num_success": 0,
                "num_fail": 1,
                "success_rate": 0.0,
                "kinematic_reachability_rate": 0.0,
                "position_fail_rate": 100.0,
                "rotation_fail_rate": 100.0,
                "theta_fail_rate": 100.0,
                "phi_fail_rate": 100.0,
                "collision_fail_rate": 100.0,
                "accuracy_fail_rate": 100.0,
                "limit_fail_rate": 100.0,
            },
        }
        return summary, data_to_save

    tip_transforms = jax.tree_util.tree_map(lambda x: x[-1:], planned_tip_traj)
    # Calculate collision mask
    vmapped_is_trajectory_in_collision = jax.vmap(
        is_trajectory_in_collision, in_axes=(0, None, None, None)
    )
    path_collision_results = vmapped_is_trajectory_in_collision(
        all_paths, robot, robot_coll, world_geom
    )
    solution_collision_mask = jnp.logical_or(path_collision_results, ~paths_are_valid)

    # Calculate final metrics
    (
        final_success_rate,
        kinematic_reachability_rate,
        final_pos_error,
        final_rot_error,
        failure_stats,
    ) = mp_metric_with_coll(
        robot,
        solution_states,
        tip_transforms,
        target_position,
        target_wxyz,
        solution_collision_mask,
    )

    # Print and return results
    print(
        f"--- {method_name_upper} (Opt: {whether_use_trajopt_after_planner}) With Collision Results ---"
    )
    print(
        f"Kinematic Reachability Rate (accurate AND within limits): {kinematic_reachability_rate:.2f}%"
    )
    print(
        f"Final Success Rate (accurate, within limits, AND collision-free): {final_success_rate:.2f}%"
    )
    print(f"Final Position Error: {final_pos_error:.3f}m")
    print(f"Final Rotation Error: {final_rot_error:.3f}rad")
    print(
        f"finish solve {planner_name} (Opt: {whether_use_trajopt_after_planner}) of num sections {robot.config.num_sections}, total time: {total_time}s"
    )
    print(failure_stats)

    summary = {
        "method": method_name_upper,
        "with_optimization": whether_use_trajopt_after_planner,
        "eval num": 1,
        "actual eval num": 1,
        "num sections": robot.config.num_sections,
        "kinematic_reachability_rate": kinematic_reachability_rate,
        "position error": final_pos_error,
        "rotation error": final_rot_error,
        "success rate": final_success_rate,
        "total time": total_time,
        "failure_stats": failure_stats,
    }
    return summary, data_to_save


def eval_mp_all_sections(
    robot_type: str,
    robot_config_path: str,
    world_config_path: str,
    section_list: list,
    eval_num: int,
    save_dir: str,
    min_sample_dist_ratio: float,
    opt_after_planner: bool = False,
    planner_type: str = "trajopt",
    start_from_initialization: bool = False,
) -> list[dict]:
    all_results_summary = []

    for num_sections in section_list:
        # load robot config
        config = json.load(open(robot_config_path))
        config["num_sections"] = num_sections

        if robot_type == "cc":
            robot = CCRobot.from_config(config)
            robot_coll = RobotCollision.from_config(config)
        elif robot_type == "tdcr":
            robot = TDCRRobot.from_config(config)
            robot_coll = RobotCollision.from_config(config)

        robot_total_length = robot.config.length * robot.config.num_sections
        # load world config
        world_coll = WorldCollision.from_config(world_config_path)

        world_geom_list = world_coll.collision_geoms_no_ground

        # Set up trajopt parameters
        timesteps = 100

        # Initialize trajectory optimizer options
        traj_options = TrajOptimizerOptions()

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

        batched_fk = jax.jit(jax.vmap(robot._forward_kinematics))

        # Initialize RRT solver
        rrt_options = RRTOptions(
            batch_size=100,
            max_iterations=1000,
        )
        rrt_traj_solver = OptimizedRRT(robot, robot_coll, rrt_options)

        # Initialize PRM solver
        prm_options = PRMOptions(
            max_planning_attempts=3, batch_size=2000, parallel_edge_checks=200
        )
        prm_traj_solver = ParallelPRM(robot, robot_coll, prm_options)
        road_map_nodes = 1000
        roadmap_path = os.path.join(
            save_dir,
            "roadmaps",
            f"roadmap_section_{num_sections}_node_{road_map_nodes}.pkl",
        )
        os.makedirs(os.path.dirname(roadmap_path), exist_ok=True)
        if os.path.exists(roadmap_path):
            print(f"Loading existing roadmap: {roadmap_path}...")
            prm_traj_solver.load_roadmap(roadmap_path)
        else:
            print(f"Building new roadmap: {roadmap_path}...")
            prm_traj_solver.build_roadmap(road_map_nodes, world_geom_list)
            prm_traj_solver.save_roadmap(roadmap_path)

        print("Init done")

        planner_map = {
            "trajopt": (
                _solve_with_trajopt,
                (start_end_interpolate_jit, traj_opt, traj_options, robot_type),
            ),
            "prm": (_solve_with_prm, (start_end_ik_solver, prm_traj_solver)),
            "rrt": (_solve_with_rrt, (start_end_ik_solver, rrt_traj_solver)),
        }
        if planner_type not in planner_map:
            raise ValueError(
                f"Unknown planner_type: '{planner_type}'. Available options are {list(planner_map.keys())}"
            )

        solve_fn, solver = planner_map[planner_type]
        traj_opt_solver_methods = (traj_opt, traj_options)

        # # Warm-up call
        print(f"Warming up {planner_type.upper()} solver...")
        # Create some dummy data for warm-up
        warm_up_num = 20
        temp_dir = os.path.join(
            save_dir,
            "temp",
            f"sections_{num_sections}_eval_{warm_up_num}_start_init_{start_from_initialization}.npz",
        )
        os.makedirs(os.path.dirname(temp_dir), exist_ok=True)
        dummy_start_states, dummy_end_states = sample_collision_free_start_end_states(
            robot=robot,
            eval_num=warm_up_num,
            robot_coll=robot_coll,
            world_geom=world_geom_list,
            batched_fk=batched_fk,
            min_distance=robot_total_length * min_sample_dist_ratio,
            save_load_path=temp_dir,
            start_from_initialization=start_from_initialization,
        )

        for i in range(dummy_start_states.theta.shape[0]):
            print(f"  Warm-up iteration {i+1}/{dummy_start_states.theta.shape[0]}...")
            dummy_start = jax.tree_util.tree_map(
                lambda x: x[i : i + 1], dummy_start_states
            )
            dummy_end = jax.tree_util.tree_map(lambda x: x[i : i + 1], dummy_end_states)

            dummy_start_fk = batched_fk(dummy_start)
            dummy_end_fk = batched_fk(dummy_end)
            s_pos = jaxlie.SE3.from_matrix(dummy_start_fk[0, -1]).translation()
            s_wxyz = jaxlie.SE3.from_matrix(dummy_start_fk[0, -1]).rotation().wxyz
            e_pos = jaxlie.SE3.from_matrix(dummy_end_fk[0, -1]).translation()
            e_wxyz = jaxlie.SE3.from_matrix(dummy_end_fk[0, -1]).rotation().wxyz
            path_cfg = solve_fn(solver, s_pos, s_wxyz, e_pos, e_wxyz, world_geom_list)
            jax.block_until_ready(path_cfg)

            if (
                (planner_type == "prm" or planner_type == "rrt")
                and opt_after_planner
                and path_cfg is not None
                and path_cfg.theta.shape[0] > 0
            ):
                path_cfg = resample_trajectory(path_cfg, timesteps)
                traj_opt, traj_options = traj_opt_solver_methods
                # Warm-up for the optimization step as well
                _ = _traj_optimize(
                    robot_type, traj_opt, traj_options, world_geom_list, path_cfg
                )
                jax.block_until_ready(_)
                print(f"Finish the warm up of [{planner_type.upper()}] with TrajOpt...")
                break  # Exit after one successful warm-up with optimization
            elif path_cfg is not None and path_cfg.theta.shape[0] > 0:
                # If not using optimizer, one successful planning is enough for warm-up
                print(
                    f"Finish the warm up of [{planner_type.upper()}] without TrajOpt..."
                )
                break

        print("Warm-up complete.")

        # Sample all pairs first
        print(
            f"\n--- Sampling {eval_num} pairs for {planner_type.upper()} with {num_sections} sections ---"
        )
        sample_data_path = f"{save_dir}/sampled_states/sections_{num_sections}_eval_{eval_num}_start_init_{start_from_initialization}.npz"
        start_states, end_states = sample_collision_free_start_end_states(
            robot=robot,
            eval_num=eval_num,
            robot_coll=robot_coll,
            world_geom=world_geom_list,
            batched_fk=batched_fk,
            min_distance=robot_total_length * min_sample_dist_ratio,
            save_load_path=sample_data_path,
            start_from_initialization=start_from_initialization,
        )

        actual_eval_num = start_states.theta.shape[0]
        if actual_eval_num < eval_num:
            print(
                f"[Warning] Could only sample {actual_eval_num}/{eval_num} pairs. Continuing with fewer pairs."
            )
        if actual_eval_num == 0:
            print("Failed to sample any valid start/end pairs. Skipping evaluation.")
            continue

        trial_results = []
        all_trials_data = []
        for i in range(actual_eval_num):
            print(
                f"\n--- Evaluating Pair {i+1}/{actual_eval_num} for {planner_type.upper()}(optimize:{opt_after_planner}) with {num_sections} sections [{save_dir}]---"
            )

            # Select the i-th start and end state from the sampled pairs
            start_state_i = jax.tree_util.tree_map(lambda x: x[i : i + 1], start_states)
            end_state_i = jax.tree_util.tree_map(lambda x: x[i : i + 1], end_states)

            summary, trial_data = _eval_planner(
                planner_name=planner_type,
                robot_type=robot_type,
                solver=solver,
                solve_fn=solve_fn,
                target_timesteps=timesteps,
                whether_use_trajopt_after_planner=opt_after_planner,
                traj_opt_solver=traj_opt_solver_methods,
                robot=robot,
                batched_fk=batched_fk,
                robot_coll=robot_coll,
                world_geom=world_geom_list,
                start_states=start_state_i,
                end_states=end_state_i,
            )
            trial_results.append(summary)
            if trial_data:
                trial_data["trial_id"] = i
                all_trials_data.append(trial_data)

        if not trial_results:
            continue

        # Extract metrics from the list of result dictionaries
        success_rates = np.array([r["success rate"] for r in trial_results])
        kinematic_rates = np.array(
            [r["kinematic_reachability_rate"] for r in trial_results]
        )
        pos_errors = np.array([r["position error"] for r in trial_results])
        rot_errors = np.array([r["rotation error"] for r in trial_results])
        times = np.array([r["total time"] for r in trial_results])

        # Save the collected raw results to a single npz file for this configuration
        full_results_path = f"{save_dir}/all_trials_results/{planner_type}_opt_{opt_after_planner}_sections_{num_sections}_all_trials_results.npz"
        os.makedirs(os.path.dirname(full_results_path), exist_ok=True)
        np.savez(
            full_results_path,
            success_rates=success_rates,
            kinematic_rates=kinematic_rates,
            pos_errors=pos_errors,
            rot_errors=rot_errors,
            times=times,
            failure_stats=[r["failure_stats"] for r in trial_results],
        )
        print(
            f"\nSaved full results for {planner_type.upper()} with {opt_after_planner} optimization and {num_sections} sections to {full_results_path}"
        )

        # Save all successful trial trajectory data
        if all_trials_data:
            aggregated_data_for_npz = {}
            for data in all_trials_data:
                trial_id = data.pop("trial_id")
                for key, value in data.items():
                    aggregated_data_for_npz[f"trial_{trial_id}_{key}"] = value

            full_trajectory_data_path = f"{save_dir}/all_trials_trajectories/{planner_type}_opt_{opt_after_planner}_sections_{num_sections}_all_trials_trajectories.npz"
            os.makedirs(os.path.dirname(full_trajectory_data_path), exist_ok=True)
            np.savez(full_trajectory_data_path, **aggregated_data_for_npz)
            print(
                f"\nSaved full trajectory data for {planner_type.upper()} with {opt_after_planner} optimization and {num_sections} sections to {full_trajectory_data_path}"
            )

        # Create the aggregated summary dictionary
        summary = {
            "method": planner_type.upper(),
            "with_optimization": opt_after_planner,
            "num sections": num_sections,
            "eval num": actual_eval_num,
            "success_rate_mean": np.mean(success_rates),
            "success_rate_std": np.std(success_rates),
            "kinematic_rate_mean": np.mean(kinematic_rates),
            "kinematic_rate_std": np.std(kinematic_rates),
            "pos_error_mean": np.mean(pos_errors),
            "pos_error_std": np.std(pos_errors),
            "rot_error_mean": np.mean(rot_errors),
            "rot_error_std": np.std(rot_errors),
            "time_mean": np.mean(times),
            "time_std": np.std(times),
        }
        all_results_summary.append(summary)

    return all_results_summary


def eval_prm_opt_time(
    robot_type: str,
    robot_config_path: str,
    world_config_path: str,
    section_list: list,
    eval_num: int,
    save_dir: str,
    min_sample_dist_ratio: float,
) -> list[dict]:
    """
    Evaluates the time taken by each component of the PRM+TrajOpt pipeline.
    (IK, PRM planning, Trajectory Optimization).
    """
    all_results_summary = []

    for num_sections in section_list:
        # Load robot config
        config = json.load(open(robot_config_path))
        config["num_sections"] = num_sections

        if robot_type == "cc":
            robot = CCRobot.from_config(config)
            robot_coll = RobotCollision.from_config(config)
        elif robot_type == "tdcr":
            robot = TDCRRobot.from_config(config)
            robot_coll = RobotCollision.from_config(config)

        robot_total_length = robot.config.length * robot.config.num_sections
        # Load world config
        world_coll = WorldCollision.from_config(world_config_path)
        world_geom_list = world_coll.collision_geoms_no_ground

        # Set up trajopt parameters
        timesteps = 100
        traj_options = TrajOptimizerOptions()
        traj_solver = TrajOptimizer(robot, robot_coll, timesteps, options=traj_options)
        start_end_ik_solver = traj_solver._ik_solver_best
        if robot_type == "cc":
            traj_opt = jax.jit(traj_solver.optimize)
        elif robot_type == "tdcr":
            traj_opt = jax.jit(traj_solver.optimize_tdcr)

        traj_opt_solver_methods = (traj_opt, traj_options)
        batched_fk = jax.jit(jax.vmap(robot._forward_kinematics))

        # Initialize PRM solver
        prm_options = PRMOptions(batch_size=2000, parallel_edge_checks=200)
        prm_traj_solver = ParallelPRM(robot, robot_coll, prm_options)
        roadmap_path = os.path.join(save_dir, "roadmaps", f"roadmap_{num_sections}.pkl")
        os.makedirs(os.path.dirname(roadmap_path), exist_ok=True)
        if os.path.exists(roadmap_path):
            print(f"Loading existing roadmap for time test: {roadmap_path}...")
            prm_traj_solver.load_roadmap(roadmap_path)
        else:
            print(f"Building new roadmap for time test: {roadmap_path}...")
            prm_traj_solver.build_roadmap(1000, world_geom_list)
            prm_traj_solver.save_roadmap(roadmap_path)

        prm_solver_tuple = (start_end_ik_solver, prm_traj_solver)

        # Warm-up
        print(f"Warming up PRM+Opt time test for {num_sections} sections...")
        warm_up_num = 5
        temp_dir = os.path.join(
            save_dir,
            "temp",
            f"time_test_sections_{num_sections}_eval_{warm_up_num}.npz",
        )
        dummy_start_states, dummy_end_states = sample_collision_free_start_end_states(
            robot=robot,
            eval_num=warm_up_num,
            robot_coll=robot_coll,
            world_geom=world_geom_list,
            batched_fk=batched_fk,
            min_distance=robot_total_length * min_sample_dist_ratio,
            save_load_path=temp_dir,
        )

        for i in range(min(warm_up_num, dummy_start_states.theta.shape[0])):
            print(f"  Warm-up iteration {i+1}/{warm_up_num}...")
            dummy_start = jax.tree_util.tree_map(
                lambda x: x[i : i + 1], dummy_start_states
            )
            dummy_end = jax.tree_util.tree_map(lambda x: x[i : i + 1], dummy_end_states)

            s_fk = batched_fk(dummy_start)
            e_fk = batched_fk(dummy_end)
            s_pos = jaxlie.SE3.from_matrix(s_fk[0, -1]).translation()
            s_wxyz = jaxlie.SE3.from_matrix(s_fk[0, -1]).rotation().wxyz
            e_pos = jaxlie.SE3.from_matrix(e_fk[0, -1]).translation()
            e_wxyz = jaxlie.SE3.from_matrix(e_fk[0, -1]).rotation().wxyz

            _, _, opt_time = test_time_of_prm_opt(
                robot_type,
                prm_solver_tuple,
                traj_opt_solver_methods,
                timesteps,
                s_pos,
                s_wxyz,
                e_pos,
                e_wxyz,
                world_geom_list,
            )
            if not np.isnan(opt_time):
                print("Warm-up successful.")
                break
        print("Warm-up complete.")

        # Sample all pairs for evaluation
        sample_data_path = f"{save_dir}/sampled_states/time_test_sections_{num_sections}_eval_{eval_num}.npz"
        os.makedirs(os.path.dirname(sample_data_path), exist_ok=True)
        start_states, end_states = sample_collision_free_start_end_states(
            robot=robot,
            eval_num=eval_num,
            robot_coll=robot_coll,
            world_geom=world_geom_list,
            batched_fk=batched_fk,
            min_distance=robot_total_length * min_sample_dist_ratio,
            save_load_path=sample_data_path,
        )

        actual_eval_num = start_states.theta.shape[0]
        if actual_eval_num == 0:
            print("Failed to sample any valid start/end pairs. Skipping time test.")
            continue

        # Run evaluation
        ik_times, prm_times, opt_times = [], [], []
        for i in range(actual_eval_num):
            print(
                f"\n--- Timing Pair {i+1}/{actual_eval_num} for {num_sections} sections ---"
            )
            start_state_i = jax.tree_util.tree_map(lambda x: x[i : i + 1], start_states)
            end_state_i = jax.tree_util.tree_map(lambda x: x[i : i + 1], end_states)

            s_fk = batched_fk(start_state_i)
            e_fk = batched_fk(end_state_i)
            s_pos = jaxlie.SE3.from_matrix(s_fk[0, -1]).translation()
            s_wxyz = jaxlie.SE3.from_matrix(s_fk[0, -1]).rotation().wxyz
            e_pos = jaxlie.SE3.from_matrix(e_fk[0, -1]).translation()
            e_wxyz = jaxlie.SE3.from_matrix(e_fk[0, -1]).rotation().wxyz

            ik_time, prm_time, opt_time = test_time_of_prm_opt(
                robot_type,
                prm_solver_tuple,
                traj_opt_solver_methods,
                timesteps,
                s_pos,
                s_wxyz,
                e_pos,
                e_wxyz,
                world_geom_list,
            )
            if not np.isnan(opt_time):
                ik_times.append(ik_time)
                prm_times.append(prm_time)
                opt_times.append(opt_time)

        # Save and summarize results
        if not ik_times:
            print(
                f"No successful paths found for {num_sections} sections. No results to save."
            )
            continue

        time_results_path = f"{save_dir}/time_test_result/prm_opt_time_results_sections_{num_sections}.npz"
        os.makedirs(os.path.dirname(time_results_path), exist_ok=True)
        np.savez(
            time_results_path,
            ik_times=np.array(ik_times),
            prm_times=np.array(prm_times),
            opt_times=np.array(opt_times),
        )
        print(f"Saved time results to {time_results_path}")

        if len(ik_times) < 200:
            print(
                f"Insufficient IK times collected for {num_sections} sections. Collected: {len(ik_times)}"
            )
            print("Use all for evaluation.")
        else:
            print("Use 200 samples for evaluation.")
            ik_times = ik_times[:200]
            prm_times = prm_times[:200]
            opt_times = opt_times[:200]

        summary = {
            "num_sections": num_sections,
            "eval_num": len(ik_times),
            "ik_time_mean": np.mean(ik_times),
            "ik_time_std": np.std(ik_times),
            "prm_time_mean": np.mean(prm_times),
            "prm_time_std": np.std(prm_times),
            "opt_time_mean": np.mean(opt_times),
            "opt_time_std": np.std(opt_times),
            "total_time_mean": np.mean(
                np.array(ik_times) + np.array(prm_times) + np.array(opt_times)
            ),
            "total_time_std": np.std(
                np.array(ik_times) + np.array(prm_times) + np.array(opt_times)
            ),
        }
        all_results_summary.append(summary)
        # Print summary with keys formatted and values in ms
        summary_str = f"Summary for {num_sections} sections:\n"
        for k, v in summary.items():
            if isinstance(v, float) or isinstance(v, np.floating):
                key_str = k.replace("_", " ").title() + " (ms)"
                val_str = f"{v * 1000:.2f}"
                summary_str += f"  {key_str}: {val_str}\n"
            else:
                key_str = k.replace("_", " ").title()
                summary_str += f"  {key_str}: {v}\n"
        print(summary_str)

    return all_results_summary


if __name__ == "__main__":
    robot_type = "tdcr"
    test_list = [3, 4, 5, 6]
    repeat_num = 200  # Evaluate 50 times for each configuration
    robot_config_path = "configs/robots/cc_scene_eval_tdcr.json"

    start_from_initialization = True

    # Correctly format the world config paths
    world_config_paths = [
        f"configs/maps/mp_scene/obstacles_random_section_{i}.json" for i in test_list
    ] + [
        "configs/maps/mp_scene/13.pick_from_shelf.json",
        "configs/maps/mp_scene/14.pick_from_bookshelf.json",
        "configs/maps/mp_scene/15.grab_from_box.json",
        "configs/maps/mp_scene/mp_demo.json",
    ]
    world_config_paths = [
        "configs/maps/mp_scene/obstacles_random_start_init_True_section_3.json"
    ]

    planner_types = ["rrt"]  # ["trajopt", "prm", "rrt"]
    opt_options = [True, False]
    result_summarys = []

    for world_path in world_config_paths:
        # Extract scene name for result directory
        scene_name = os.path.splitext(os.path.basename(world_path))[0]
        result_dir = f"results/debug_draw/{scene_name}"
        os.makedirs(result_dir, exist_ok=True)

        print(f"\n{'='*20} Running Evaluation for Scene: {scene_name} {'='*20}")

        for planner in planner_types:
            # TrajOpt doesn't have a separate optimization option like PRM/RRT
            if planner == "trajopt":
                opts_to_run = [False]  # It ignores this flag, just run once
            else:
                opts_to_run = opt_options

            for opt in opts_to_run:
                print(f"\n--- Planner: {planner.upper()}, Optimization: {opt} ---")

                current_test_list = test_list
                if "obstacles_random_section" in world_path:
                    # Extract number of sections from filename
                    section_num = int(scene_name.split("obstacles_random_section_")[1])
                    current_test_list = [section_num]

                result_summary = eval_mp_all_sections(
                    robot_config_path=robot_config_path,
                    world_config_path=world_path,
                    section_list=current_test_list,
                    eval_num=repeat_num,
                    save_dir=result_dir,
                    min_sample_dist_ratio=0.05,
                    opt_after_planner=opt,
                    robot_type=robot_type,
                    planner_type=planner,
                    start_from_initialization=start_from_initialization,
                )
                if result_summary:
                    result_summarys.extend(result_summary)
                csv_path = os.path.join(
                    result_dir, "analysis", "all_trials_detailed.csv"
                )
                save_results(results_dir=result_dir, detailed_csv_path=csv_path)

    print("\n\n" + "=" * 30 + " FINAL SUMMARY " + "=" * 30)
    summarize_results(result_summarys)

    prm_test_world = "configs/maps/mp_scene/obstacles_random_section_3.json"
    prm_section_list = [3]
    repeat_num_prm = 230
    scene_name = os.path.splitext(os.path.basename(prm_test_world))[0]
    result_dir = f"results/mp_test/{scene_name}"
    os.makedirs(result_dir, exist_ok=True)
    result_summary = eval_prm_opt_time(
        robot_type=robot_type,
        robot_config_path=robot_config_path,
        world_config_path=prm_test_world,
        section_list=prm_section_list,
        eval_num=repeat_num_prm,
        save_dir=result_dir,
        min_sample_dist_ratio=0.05,
    )
    print("\n\n" + "=" * 30 + " PRM OPT TIME SUMMARY " + "=" * 30)
    print(result_summary)
