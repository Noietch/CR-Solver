import argparse
import os
import time
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

# Initialize JAX persistent compilation cache
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir"
)
from jax.experimental.compilation_cache import compilation_cache as cc

cc.set_cache_dir("/tmp/jax_cache")

from benchmark.mp.mp_analyze import save_log_to_csv
from benchmark.mp.problems import Problem
from benchmark.mp.utils import log_result, save_result

from cr_solver.geom import CollGeom, RobotCollision, WorldCollision
from cr_solver.robots.cc_robot import ConstantCurvatureState
from cr_solver.robots.tdcr_robot import TDCRRobot
from cr_solver.solver import (
    OptimizedRRT,
    ParallelPRM,
    PRMOptions,
    RRTOptions,
    TrajOptimizer,
    TrajOptimizerOptions,
)
from cr_solver.solver.motion_planner import resample_trajectory

jax.config.update("jax_default_matmul_precision", "highest")

DISABLE_JIT = False

if DISABLE_JIT:
    import os

    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def init_solver(
    robot: TDCRRobot,
    robot_coll: RobotCollision,
    world_geom_list: Sequence[CollGeom],
    save_dir: str,
    num_sections: int,
    road_map_nodes: int,
    timesteps: int = 100,
    max_iter_num: int = 3,
):
    traj_options = TrajOptimizerOptions(max_iter_num=max_iter_num)
    traj_solver = TrajOptimizer(
        robot, robot_coll, timesteps, options=traj_options
    )
    traj_opt = jax.jit(traj_solver.optimize_tdcr)

    # Initialize RRT solver
    rrt_options = RRTOptions(
        batch_size=100,
        max_iterations=1000,
    )
    rrt_traj_solver = OptimizedRRT(robot, robot_coll, rrt_options)

    # Initialize PRM solver
    prm_options = PRMOptions(batch_size=2000, parallel_edge_checks=200)
    prm_traj_solver = ParallelPRM(robot, robot_coll, prm_options)

    roadmap_path = os.path.join(
        save_dir,
        "roadmaps",
        f"roadmap_section_{num_sections}_node_{road_map_nodes}.pkl",
    )
    os.makedirs(os.path.dirname(roadmap_path), exist_ok=True)
    if os.path.exists(roadmap_path):
        print("Loading existing roadmap...")
        prm_traj_solver.load_roadmap(roadmap_path)
    else:
        print("Building new roadmap...")
        prm_traj_solver.build_roadmap(road_map_nodes, world_geom_list)
        prm_traj_solver.save_roadmap(roadmap_path)
    print("Init done")

    return (
        traj_options,
        traj_solver,
        traj_opt,
        rrt_traj_solver,
        prm_traj_solver,
    )


def _traj_optimize(
    traj_opt: Callable,
    traj_options: TrajOptimizerOptions,
    world_coll: Sequence[CollGeom],
    cfg: Array,
) -> ConstantCurvatureState:
    """Optimizes a trajectory using the TrajOpt framework."""
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


def solve_with_trajopt(
    start_state: ConstantCurvatureState,
    end_state: ConstantCurvatureState,
    world_coll: Sequence[CollGeom],
    traj_solver: TrajOptimizer,
    traj_opt: callable,
    traj_options: TrajOptimizerOptions,
) -> tuple[ConstantCurvatureState, float]:
    start_time = time.time()
    cfg = traj_solver.find_path(start_state, end_state)
    path_cfg = _traj_optimize(traj_opt, traj_options, world_coll, cfg)
    jax.block_until_ready(path_cfg)
    trajopt_time = time.time() - start_time
    return path_cfg, trajopt_time


def solve_with_prm(
    start_state: ConstantCurvatureState,
    end_state: ConstantCurvatureState,
    world_coll: Sequence[CollGeom],
    prm_traj_solver: ParallelPRM,
    traj_opt: callable,
    traj_options: TrajOptimizerOptions,
    with_opt: bool,
    target_timesteps: int,
):
    start_prm_time = time.time()
    path_cfg = prm_traj_solver.find_path(start_state, end_state, world_coll)
    prm_time = time.time() - start_prm_time

    opt_time = 0.0
    if path_cfg is None:
        print("No path found")
        return None, prm_time, opt_time

    if with_opt and path_cfg.theta.shape[0] > 0:
        start_opt_time = time.time()
        cfg = resample_trajectory(path_cfg, target_timesteps)
        path_cfg = _traj_optimize(traj_opt, traj_options, world_coll, cfg)
        opt_time = time.time() - start_opt_time

    return path_cfg, prm_time, opt_time


def solve_with_rrt(
    start_state: ConstantCurvatureState,
    end_state: ConstantCurvatureState,
    world_coll: Sequence[CollGeom],
    rrt_traj_solver: OptimizedRRT,
    traj_opt: callable,
    traj_options: TrajOptimizerOptions,
    with_opt: bool,
    target_timesteps: int,
) -> tuple[ConstantCurvatureState, float]:
    start_time = time.time()
    path_cfg = rrt_traj_solver.find_path(start_state, end_state, world_coll)
    rrt_time = time.time() - start_time
    if path_cfg is None:
        print("No path found")
        return None, rrt_time
    if with_opt and path_cfg.theta.shape[0] > 0:
        path_cfg = resample_trajectory(path_cfg, target_timesteps)
        path_cfg = _traj_optimize(traj_opt, traj_options, world_coll, path_cfg)
        jax.block_until_ready(path_cfg)
        rrt_time = time.time() - start_time
    return path_cfg, rrt_time


def eval_mp_with_coll_scene(
    robot_config_path: str,
    world_config_path: str,
    save_dir: str,
    num_sections: int,
    road_map_nodes: int,
    eval_num: int,
    start_from_initialization: bool,
    remove_failed_trials: bool,
    run_after_filtered: bool,
    min_sample_dist_ratio: float,
    max_iter_num: int,
):
    robot = TDCRRobot.from_config(robot_config_path)
    robot_coll = RobotCollision.from_config(robot_config_path)
    world_coll = WorldCollision.from_config(world_config_path)
    world_geom_list = world_coll.collision_geoms_no_ground
    robot_total_length = robot.config.length * robot.config.num_sections

    timesteps = 100
    (
        traj_options,
        traj_solver,
        traj_opt,
        rrt_traj_solver,
        prm_traj_solver,
    ) = init_solver(
        robot=robot,
        robot_coll=robot_coll,
        world_geom_list=world_geom_list,
        save_dir=save_dir,
        num_sections=num_sections,
        road_map_nodes=road_map_nodes,
        timesteps=timesteps,
        max_iter_num=max_iter_num,
    )

    batched_fk = jax.jit(jax.vmap(robot._forward_kinematics))

    # Load problems
    print(
        f"\n--- Sampling {eval_num} pairs with {num_sections} sections "
        f"and start initialization {start_from_initialization} ---"
    )
    if run_after_filtered:
        rename_suffix = "_prm_success"
    else:
        rename_suffix = ""
    sample_data_path = (
        f"{save_dir}/sampled_states/sections_{num_sections}_eval_"
        f"{eval_num}_start_init_{start_from_initialization}"
        f"{rename_suffix}.npz"
    )
    problem = Problem(
        sample_data_path=sample_data_path,
        eval_num=eval_num,
        robot=robot,
        robot_coll=robot_coll,
        world_geom=world_geom_list,
        batched_fk=batched_fk,
        min_distance=robot_total_length * min_sample_dist_ratio,
        start_from_initialization=start_from_initialization,
    )
    start_states, end_states = problem.load(sample_data_path)

    # Warm up
    for i in range(3):
        print(f"Start warm up for {i+1}/3")
        start_state = jax.tree_util.tree_map(
            lambda x: x[i:i + 1], start_states
        )
        end_state = jax.tree_util.tree_map(lambda x: x[i:i + 1], end_states)
        # Squeeze the batch dimension from start and end states
        start_state_i = jax.tree_util.tree_map(
            lambda x: jnp.squeeze(x, axis=0), start_state
        )
        end_state_i = jax.tree_util.tree_map(
            lambda x: jnp.squeeze(x, axis=0), end_state
        )
        solve_with_trajopt(
            start_state_i,
            end_state_i,
            world_geom_list,
            traj_solver,
            traj_opt,
            traj_options,
        )
        solve_with_prm(
            start_state_i,
            end_state_i,
            world_geom_list,
            prm_traj_solver,
            traj_opt,
            traj_options,
            True,
            timesteps,
        )
        solve_with_rrt(
            start_state_i,
            end_state_i,
            world_geom_list,
            rrt_traj_solver,
            traj_opt,
            traj_options,
            True,
            timesteps,
        )
    print("Finished warm up....")

    actual_eval_num = start_states.theta.shape[0]
    trajopt_success = []
    traj_opt_total_time = []
    traj_opt_traj_length = []

    prm_success = []
    prm_total_time = []
    prm_traj_length = []
    prm_opt_success = []
    prm_opt_total_time = []
    prm_opt_traj_length = []

    rrt_success = []
    rrt_total_time = []
    rrt_traj_length = []
    rrt_opt_success = []
    rrt_opt_total_time = []
    rrt_opt_traj_length = []
    all_trials_data = []

    for i in range(actual_eval_num):
        print(
            f"\n--- Evaluating Pair {i+1}/{actual_eval_num} with "
            f"{num_sections} sections [{save_dir}]---"
        )

        # Select the i-th start and end state from the sampled pairs
        start_state = jax.tree_util.tree_map(
            lambda x: x[i:i + 1], start_states
        )
        end_state = jax.tree_util.tree_map(lambda x: x[i:i + 1], end_states)
        # Squeeze the batch dimension from start and end states
        start_state_i = jax.tree_util.tree_map(
            lambda x: jnp.squeeze(x, axis=0), start_state
        )
        end_state_i = jax.tree_util.tree_map(
            lambda x: jnp.squeeze(x, axis=0), end_state
        )
        base_info = (
            start_state_i,
            end_state_i,
            robot,
            robot_coll,
            world_geom_list,
            batched_fk,
            i,
        )

        path_cfg, trajopt_time = solve_with_trajopt(
            start_state=start_state_i,
            end_state=end_state_i,
            world_coll=world_geom_list,
            traj_solver=traj_solver,
            traj_opt=traj_opt,
            traj_options=traj_options,
        )
        data_to_save, successful_id, total_time, traj_length = log_result(
            base_info=base_info,
            path_cfg=path_cfg,
            method_name="TRAJOPT",
            trajopt_time=trajopt_time,
        )
        all_trials_data.append(data_to_save)
        trajopt_success.append(successful_id)
        traj_opt_total_time.append(total_time)
        traj_opt_traj_length.append(traj_length)

        path_cfg, prm_time, opt_time = solve_with_prm(
            start_state=start_state_i,
            end_state=end_state_i,
            world_coll=world_geom_list,
            prm_traj_solver=prm_traj_solver,
            traj_opt=traj_opt,
            traj_options=traj_options,
            with_opt=False,
            target_timesteps=timesteps,
        )
        data_to_save, successful_id, total_time, traj_length = log_result(
            base_info=base_info,
            path_cfg=path_cfg,
            method_name="PRM",
            prm_time=prm_time,
            opt_time=opt_time,
            with_opt=False,
        )
        all_trials_data.append(data_to_save)
        prm_success.append(successful_id)
        prm_total_time.append(total_time)
        prm_traj_length.append(traj_length)

        path_cfg, prm_time, opt_time = solve_with_prm(
            start_state=start_state_i,
            end_state=end_state_i,
            world_coll=world_geom_list,
            prm_traj_solver=prm_traj_solver,
            traj_opt=traj_opt,
            traj_options=traj_options,
            with_opt=True,
            target_timesteps=timesteps,
        )
        data_to_save, successful_id, total_time, traj_length = log_result(
            base_info=base_info,
            path_cfg=path_cfg,
            method_name="PRM",
            prm_time=prm_time,
            opt_time=opt_time,
            with_opt=True,
        )
        all_trials_data.append(data_to_save)
        prm_opt_success.append(successful_id)
        prm_opt_total_time.append(total_time)
        prm_opt_traj_length.append(traj_length)

        path_cfg, rrt_time = solve_with_rrt(
            start_state=start_state_i,
            end_state=end_state_i,
            world_coll=world_geom_list,
            rrt_traj_solver=rrt_traj_solver,
            traj_opt=traj_opt,
            traj_options=traj_options,
            with_opt=False,
            target_timesteps=timesteps,
        )
        data_to_save, successful_id, total_time, traj_length = log_result(
            base_info=base_info,
            path_cfg=path_cfg,
            method_name="RRT",
            rrt_time=rrt_time,
            with_opt=False,
        )
        all_trials_data.append(data_to_save)
        rrt_success.append(successful_id)
        rrt_total_time.append(total_time)
        rrt_traj_length.append(traj_length)

        path_cfg, rrt_time = solve_with_rrt(
            start_state=start_state_i,
            end_state=end_state_i,
            world_coll=world_geom_list,
            rrt_traj_solver=rrt_traj_solver,
            traj_opt=traj_opt,
            traj_options=traj_options,
            with_opt=True,
            target_timesteps=timesteps,
        )
        data_to_save, successful_id, total_time, traj_length = log_result(
            base_info=base_info,
            path_cfg=path_cfg,
            method_name="RRT",
            rrt_time=rrt_time,
            with_opt=True,
        )
        all_trials_data.append(data_to_save)
        rrt_opt_success.append(successful_id)
        rrt_opt_total_time.append(total_time)
        rrt_opt_traj_length.append(traj_length)

    save_result(
        save_dir=save_dir,
        world_config_path=world_config_path,
        eval_num=eval_num,
        actual_eval_num=actual_eval_num,
        num_sections=num_sections,
        road_map_nodes=road_map_nodes,
        trajopt_success=trajopt_success,
        traj_opt_total_time=traj_opt_total_time,
        traj_opt_traj_length=traj_opt_traj_length,
        prm_success=prm_success,
        prm_total_time=prm_total_time,
        prm_traj_length=prm_traj_length,
        prm_opt_success=prm_opt_success,
        prm_opt_total_time=prm_opt_total_time,
        prm_opt_traj_length=prm_opt_traj_length,
        rrt_success=rrt_success,
        rrt_total_time=rrt_total_time,
        rrt_traj_length=rrt_opt_traj_length,
        rrt_opt_success=rrt_opt_success,
        rrt_opt_total_time=rrt_opt_total_time,
        rrt_opt_traj_length=rrt_opt_traj_length,
    )

    full_trajectory_data_path = (
        f"{save_dir}/all_trials_trajectories/sections_{num_sections}"
        f"_all_trials_trajectories.npz"
    )
    os.makedirs(os.path.dirname(full_trajectory_data_path), exist_ok=True)
    successful_data = [d for d in all_trials_data if isinstance(d, dict) and d]
    np.savez_compressed(
        full_trajectory_data_path,
        all_trials_data=np.array(successful_data, dtype=object),
    )
    print(
        f"Saved {len(successful_data)} trial(s) to {full_trajectory_data_path}"
    )

    if remove_failed_trials:
        rename_suffix = "_prm_success"
        if run_after_filtered:
            rename_suffix = ""
        prm_success_filtered = [
            item for item in prm_success if item is not None
        ]
        problem.save(
            jnp.array(prm_success_filtered), rename_suffix=rename_suffix
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MP evaluation args")
    parser.add_argument(
        "--section-num", type=int, default=4, help="number of sections"
    )
    parser.add_argument(
        "--repeat-num",
        type=int,
        default=60,
        help="how many evaluations to run"
    )
    parser.add_argument(
        "--world-config",
        dest="world_config_path",
        type=str,
        default="configs/maps/mp_scene/mp_demo.json",
        help="path to world config json",
    )
    parser.add_argument(
        "--test-name",
        dest="test_name",
        type=str,
        default="mp_test",
        help="name of the test",
    )
    parser.add_argument(
        "--max-iter", type=int, default=3, help="maximum number of iterations"
    )
    args = parser.parse_args()

    section_num: int = args.section_num
    repeat_num: int = args.repeat_num
    world_config_path: str = args.world_config_path
    test_name: str = args.test_name
    max_iter_num: int = args.max_iter

    start_from_initialization = False
    run_after_filtered = True
    remove_failed_trials = True
    robot_config_path = "configs/robots/cc_scene_eval_tdcr.json"
    scene_name = os.path.splitext(os.path.basename(world_config_path))[0]
    result_dir = f"results/{test_name}/{scene_name}"
    print(f"\n{'='*20} Running Evaluation for Scene: {scene_name} {'='*20}")
    eval_mp_with_coll_scene(
        robot_config_path=robot_config_path,
        world_config_path=world_config_path,
        save_dir=result_dir,
        num_sections=section_num,
        road_map_nodes=1500,
        eval_num=repeat_num,
        start_from_initialization=start_from_initialization,
        remove_failed_trials=remove_failed_trials,
        run_after_filtered=run_after_filtered,
        min_sample_dist_ratio=0.1,
        max_iter_num=max_iter_num,
    )
    save_log_to_csv(f"results/{test_name}")
