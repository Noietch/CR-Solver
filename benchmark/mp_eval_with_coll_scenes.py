import jax
from jaxtyping import Array
import numpy as np
import jax.numpy as jnp
import jaxlie
import time
import json
import os
from typing import Callable, Sequence
from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.solver import RRTMotionPlanner, MotionPlanner, SamplingBasedMotionPlanner
from soul.geom import (
    RobotCollision,
    WorldCollision,
    CollGeom,
    colldist_from_sdf,
)

jax.config.update("jax_default_matmul_precision", "highest")

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def mp_metric_with_coll(
    robot: CCRobot,
    solution: ConstantCurvatureState,
    result_transform: jaxlie.SE3,
    target_position: Array,
    target_orientation: Array,
    solution_collision_mask: Array,
) -> tuple[float, float, float, float, dict]:

    result_position = result_transform.translation()
    result_orientation = result_transform.rotation()

    # Accuracy thresholds
    position_threshold: float = float("inf")
    rotation_threshold: float = float("inf")

    position_error = jnp.linalg.norm(result_position - target_position, axis=-1)
    orientation_error = jnp.linalg.norm(
        jnp.array(
            (jaxlie.SO3(target_orientation).inverse() @ result_orientation).log()
        ),
        axis=-1,
    )

    # Individual failure masks for accuracy
    position_fail_mask = position_error >= position_threshold
    rotation_fail_mask = orientation_error >= rotation_threshold
    acc_mask = jnp.logical_and(
        position_error < position_threshold,
        orientation_error < rotation_threshold,
    )

    # Check joint limit constraints
    delta = 0.01  # 0.01 is the margin for the joint limit constraint
    theta_mask = jnp.all(
        jnp.logical_and(
            solution.theta >= robot.config.lower_limits_theta - delta,
            solution.theta <= robot.config.upper_limits_theta + delta,
        ),
        axis=-1,
    )
    phi_mask = jnp.all(
        jnp.logical_and(
            solution.phi >= robot.config.lower_limits_phi - delta,
            solution.phi <= robot.config.upper_limits_phi + delta,
        ),
        axis=-1,
    )

    # Individual constraint violation masks
    theta_fail_mask = ~theta_mask
    phi_fail_mask = ~phi_mask
    joint_limits_mask = theta_mask & phi_mask

    # Combined masks
    accuracy_and_limits_mask = acc_mask & joint_limits_mask

    # Final success mask (accurate, within limits, and collision-free)
    final_success_mask = jnp.logical_and(
        accuracy_and_limits_mask,
        ~solution_collision_mask,
    )

    # Kinematic reachability (accurate and within limits, ignoring collision)
    kinematic_reachability_rate = jnp.mean(accuracy_and_limits_mask) * 100.0

    # Calculate detailed failure statistics
    total_samples = len(final_success_mask)
    num_success = jnp.sum(final_success_mask)
    num_fail = total_samples - num_success

    # Individual failure types
    num_position_fail = jnp.sum(position_fail_mask)
    num_rotation_fail = jnp.sum(rotation_fail_mask)
    num_theta_fail = jnp.sum(theta_fail_mask)
    num_phi_fail = jnp.sum(phi_fail_mask)
    num_collision_fail = jnp.sum(solution_collision_mask)

    # Combined failure categories
    num_accuracy_fail = jnp.sum(~acc_mask)  # Failed due to position OR rotation
    num_limit_fail = jnp.sum(~joint_limits_mask)  # Failed due to any joint limits

    failure_stats = {
        "total_samples": int(total_samples),
        "num_success": int(num_success),
        "num_fail": int(num_fail),
        "success_rate": float(jnp.mean(final_success_mask) * 100.0),
        "kinematic_reachability_rate": float(kinematic_reachability_rate),
        # Percentages
        "position_fail_rate": float(num_position_fail / total_samples * 100),
        "rotation_fail_rate": float(num_rotation_fail / total_samples * 100),
        "theta_fail_rate": float(num_theta_fail / total_samples * 100),
        "phi_fail_rate": float(num_phi_fail / total_samples * 100),
        "collision_fail_rate": float(num_collision_fail / total_samples * 100),
        "accuracy_fail_rate": float(num_accuracy_fail / total_samples * 100),
        "limit_fail_rate": float(num_limit_fail / total_samples * 100),
    }

    final_success_rate = jnp.mean(final_success_mask) * 100.0
    # Use jnp.nanmean to avoid errors if no solutions are successful
    final_pos_error = jnp.nan_to_num(jnp.mean(position_error[final_success_mask]))
    final_rot_error = jnp.nan_to_num(jnp.mean(orientation_error[final_success_mask]))

    return (
        final_success_rate,
        kinematic_reachability_rate,
        final_pos_error,
        final_rot_error,
        failure_stats,
    )


def sample_collision_free_start_end_states(
    robot: CCRobot,
    eval_num: int,
    robot_coll: RobotCollision,
    world_geom: CollGeom,
    batched_fk: Callable[[ConstantCurvatureState], Array],
    min_distance: float = 0.1,
    batch_size: int = 256,
    save_load_path: str = None,
) -> tuple[ConstantCurvatureState, ConstantCurvatureState]:
    """
    Samples a specified number of pairs of collision-free start and end states,
    ensuring their end-effector positions are separated by a minimum distance.

    If a `save_load_path` is provided and the file exists, it loads the states.
    Otherwise, it samples the states and saves them to the path if provided.
    """
    if save_load_path and os.path.exists(save_load_path):
        print(f"Loading pre-sampled states from {save_load_path}...")
        data = np.load(save_load_path)
        start_states = ConstantCurvatureState(
            theta=jnp.array(data["start_theta"]),
            phi=jnp.array(data["start_phi"]),
            base_position=jnp.zeros((data["start_theta"].shape[0], 3)),
        )
        end_states = ConstantCurvatureState(
            theta=jnp.array(data["end_theta"]),
            phi=jnp.array(data["end_phi"]),
            base_position=jnp.zeros((data["end_theta"].shape[0], 3)),
        )
        print(
            f"Loaded {start_states.theta.shape[0]} start/end pairs."
        )
        return start_states, end_states

    print(
        f"Sampling {eval_num} collision-free start-end pairs with min distance {min_distance}..."
    )

    is_collision_vmap = jax.vmap(is_state_in_collision, in_axes=(0, None, None, None))

    # Sample a large pool of collision-free states first.
    pool_size = eval_num * 10  # Sample more to have options for pairing
    accepted_states = []
    total_sampled = 0
    max_pool_sampling_attempts = pool_size * 3  # Safety break for pool sampling

    print(f"Sampling a pool of {pool_size} collision-free states...")
    while (
        len(accepted_states) < pool_size and total_sampled < max_pool_sampling_attempts
    ):
        candidate_states = sample_states(robot, batch_size)
        total_sampled += batch_size

        collision_mask = is_collision_vmap(
            candidate_states, robot, robot_coll, world_geom
        )
        non_colliding_states = jax.tree_util.tree_map(
            lambda x: x[~collision_mask], candidate_states
        )

        num_non_colliding = non_colliding_states.theta.shape[0]
        if num_non_colliding > 0:
            current_pool_size = sum(s.theta.shape[0] for s in accepted_states)
            needed = pool_size - current_pool_size
            if num_non_colliding > needed:
                non_colliding_states = jax.tree_util.tree_map(
                    lambda x: x[:needed], non_colliding_states
                )

            accepted_states.append(non_colliding_states)
            new_pool_size = sum(s.theta.shape[0] for s in accepted_states)
            print(f"  ... pool has {new_pool_size}/{pool_size} states")

    if not accepted_states:
        raise RuntimeError("Could not sample any collision-free states for the pool.")

    state_pool = jax.tree_util.tree_map(
        lambda *x: jnp.concatenate(x, axis=0), *accepted_states
    )

    if state_pool.theta.shape[0] < eval_num * 2:
        print(
            f"Warning: Pool size ({state_pool.theta.shape[0]}) is small. "
            f"May not find enough pairs."
        )
        if state_pool.theta.shape[0] < 2:
            raise RuntimeError(
                "Not enough collision-free states in the pool to form any pairs."
            )

    # Pre-compute FK for the entire pool
    fk_transforms_pool = batched_fk(state_pool)
    tip_transforms_pool = jaxlie.SE3.from_matrix(fk_transforms_pool[:, -1, ...])
    positions_pool = tip_transforms_pool.translation()

    # Pair states from the pool
    final_start_states_list = []
    final_end_states_list = []
    used_indices = np.zeros(state_pool.theta.shape[0], dtype=bool)
    num_available = state_pool.theta.shape[0]
    max_pairing_attempts = num_available * 10  # Safety break for pairing

    print("Pairing states...")
    for _ in range(max_pairing_attempts):
        if len(final_start_states_list) >= eval_num:
            break
        if np.sum(~used_indices) < 2:
            print("Warning: Not enough available states to form a new pair.")
            break

        # Get available indices
        available_indices = np.where(~used_indices)[0]
        # Pick two random, unique indices from the available pool
        idx1, idx2 = np.random.choice(available_indices, size=2, replace=False)

        pos1 = positions_pool[idx1]
        pos2 = positions_pool[idx2]

        distance = jnp.linalg.norm(pos1 - pos2)

        if distance >= min_distance:
            start_state = jax.tree_util.tree_map(
                lambda x: x[idx1 : idx1 + 1], state_pool
            )
            end_state = jax.tree_util.tree_map(lambda x: x[idx2 : idx2 + 1], state_pool)

            final_start_states_list.append(start_state)
            final_end_states_list.append(end_state)

            used_indices[idx1] = True
            used_indices[idx2] = True

            if (
                len(final_start_states_list) % 10 == 0
                or len(final_start_states_list) == eval_num
            ):
                print(f"  ... found {len(final_start_states_list)}/{eval_num} pairs")

    if len(final_start_states_list) < eval_num:
        print(
            f"Warning: Could only sample {len(final_start_states_list)} pairs "
            f"out of {eval_num} required."
        )
        if not final_start_states_list:
            raise RuntimeError("Could not sample any valid state pairs.")

    final_start_states = jax.tree_util.tree_map(
        lambda *x: jnp.concatenate(x, axis=0), *final_start_states_list
    )
    final_end_states = jax.tree_util.tree_map(
        lambda *x: jnp.concatenate(x, axis=0), *final_end_states_list
    )

    # Save the states if a path is provided
    if save_load_path:
        print(f"Saving sampled states to {save_load_path}...")
        os.makedirs(os.path.dirname(save_load_path), exist_ok=True)
        np.savez(
            save_load_path,
            start_theta=np.asarray(final_start_states.theta),
            start_phi=np.asarray(final_start_states.phi),
            end_theta=np.asarray(final_end_states.theta),
            end_phi=np.asarray(final_end_states.phi),
        )

    print(f"Finished sampling {final_start_states.theta.shape[0]} start/end pairs.")
    return final_start_states, final_end_states


def sample_states(robot: CCRobot, num_states: int) -> ConstantCurvatureState:
    random_key = jax.random.PRNGKey(42)
    random_key, subkey = jax.random.split(random_key)
    theta = jax.random.uniform(
        key=subkey,
        shape=(num_states, robot.config.num_sections),
        minval=robot.config.lower_limits_theta,
        maxval=robot.config.upper_limits_theta,
    )
    phi = jax.random.uniform(
        key=subkey,
        shape=(num_states, robot.config.num_sections),
        minval=robot.config.lower_limits_phi,
        maxval=robot.config.upper_limits_phi,
    )

    states = ConstantCurvatureState(
        base_position=jnp.zeros((num_states, 3)),
        theta=theta,
        phi=phi,
    )

    return states


@jax.jit
def is_state_in_collision(
    state: ConstantCurvatureState,
    robot: CCRobot,
    robot_coll: RobotCollision,
    world_geom: CollGeom,
) -> bool:
    """Check if the robot is in collision with obstacles or itself using low-level functions."""
    margin = 0.05  # Use the same margin as the solver

    # 1. Calculate world collision cost
    world_dist = robot_coll.compute_world_collision_distance(robot, state, world_geom)
    world_cost = jnp.sum(colldist_from_sdf(world_dist, margin).flatten())

    # 2. Sum costs and check for collision
    total_cost = world_cost
    return total_cost > 1e-6


@jax.jit
def is_path_in_collision(
    start_state: ConstantCurvatureState,
    end_state: ConstantCurvatureState,
    robot: CCRobot,
    robot_coll: RobotCollision,
    world_geom: CollGeom,
    num_steps: int = 10,
) -> bool:
    """Checks for collisions along a linearly interpolated path in configuration space."""

    def single_step_check(i, carry):
        is_collided = carry
        # Avoid re-checking start (i=0) and end (i=num_steps-1) if they are checked outside
        alpha = i / (num_steps - 1)
        interp_theta = (1 - alpha) * start_state.theta + alpha * end_state.theta
        interp_phi = (1 - alpha) * start_state.phi + alpha * end_state.phi

        current_state = ConstantCurvatureState(
            base_position=end_state.base_position,  # Assuming base doesn't move
            theta=interp_theta,
            phi=interp_phi,
        )

        in_collision = is_state_in_collision(
            current_state, robot, robot_coll, world_geom
        )
        return jnp.logical_or(is_collided, in_collision)

    # Check start and end states first
    start_in_collision = is_state_in_collision(
        start_state, robot, robot_coll, world_geom
    )
    end_in_collision = is_state_in_collision(end_state, robot, robot_coll, world_geom)
    initial_collision = jnp.logical_or(start_in_collision, end_in_collision)

    # Then check intermediate points
    # We scan from 1 to num_steps-1 to avoid re-checking start and end
    path_in_collision = jax.lax.fori_loop(
        1, num_steps - 1, single_step_check, initial_collision
    )

    return path_in_collision


@jax.jit
def is_trajectory_in_collision(
    trajectory: ConstantCurvatureState,
    robot: CCRobot,
    robot_coll: RobotCollision,
    world_geom: CollGeom,
) -> bool:
    """Checks if any state in a trajectory is in collision."""
    if trajectory is None or trajectory.theta.shape[0] == 0:
        return True  # No path found is a collision/failure

    # Vmap the single-state check over the trajectory timesteps
    in_collision_mask = jax.vmap(is_state_in_collision, in_axes=(0, None, None, None))(
        trajectory, robot, robot_coll, world_geom
    )
    return jnp.any(in_collision_mask)


def summarize_results(all_results_summary: list):
    """Prints a summary table of the evaluation results."""
    print("\n\n--- MP Evaluation Summary ---")
    header = (
        f"{'Method':<10} | {'Sections':<10} | {'Eval Num':<10} | "
        f"{'Success (%)':<18} | {'Reachable (%)':<20} | "
        f"{'Pos Error (m)':<20} | {'Rot Error (rad)':<20} | {'Time (s)':<20}"
    )
    print(header)
    print("-" * len(header))

    for res_item in all_results_summary:
        method_str = res_item.get("method", "MP")
        sections_str = str(res_item.get("num sections", "N/A"))
        eval_num_str = str(res_item.get("eval num", "N/A"))

        # Format statistics with mean ± std
        sr_str = f"{res_item['success_rate_mean']:.2f} ± {res_item['success_rate_std']:.2f}"
        kr_str = f"{res_item['kinematic_rate_mean']:.2f} ± {res_item['kinematic_rate_std']:.2f}"
        ps_error_str = (
            f"{res_item['pos_error_mean']:.4f} ± {res_item['pos_error_std']:.4f}"
        )
        rt_error_str = (
            f"{res_item['rot_error_mean']:.4f} ± {res_item['rot_error_std']:.4f}"
        )
        time_str = f"{res_item['time_mean']:.3f} ± {res_item['time_std']:.3f}"

        print(
            f"{method_str:<10} | {sections_str:<10} | {eval_num_str:<10} | "
            f"{sr_str:<18} | {kr_str:<20} | "
            f"{ps_error_str:<20} | {rt_error_str:<20} | {time_str:<20}"
        )


def _solve_with_trajopt(
    solver: MotionPlanner,
    start_position: Array,
    start_wxyz: Array,
    target_position: Array,
    target_wxyz: Array,
    world_geom: CollGeom,
):
    """Specific solving logic for TrajOpt."""
    # JIT compile and warm-up
    start_end_interpolate_jit = jax.jit(solver.start_end_interpolate)
    optimize_traj_jit = jax.jit(solver.optimize)

    print("Warm-up run for TrajOpt solver...")
    path_cfg = start_end_interpolate_jit(
        start_position, start_wxyz, target_position, target_wxyz, [world_geom]
    )
    path_cfg = optimize_traj_jit(path_cfg, [world_geom])
    jax.block_until_ready(path_cfg)

    # Solve
    print(f"start solve trajopt")
    start = time.time()
    path_cfg = start_end_interpolate_jit(
        start_position, start_wxyz, target_position, target_wxyz, [world_geom]
    )
    path_cfg = optimize_traj_jit(path_cfg, [world_geom])
    jax.block_until_ready(path_cfg)
    total_time = time.time() - start

    return path_cfg, total_time


def _solve_with_prm(
    solver: SamplingBasedMotionPlanner,
    start_position: Array,
    start_wxyz: Array,
    target_position: Array,
    target_wxyz: Array,
    world_geom: CollGeom,
):
    """Specific solving logic for PRM."""
    print("Warm-up run for PRM solver...")
    cfg_pair = solver._ik_solver_best(
        start_wxyz, start_position, target_wxyz, target_position, [world_geom]
    )
    path_cfg = solver.find_path(cfg_pair[0], cfg_pair[1], 100, [world_geom])
    jax.block_until_ready(path_cfg)

    print("start solve prm")
    start = time.time()
    cfg_pair = solver._ik_solver_best(
        start_wxyz, start_position, target_wxyz, target_position, [world_geom]
    )
    path_cfg = solver.find_path(cfg_pair[0], cfg_pair[1], 1000, [world_geom])
    jax.block_until_ready(path_cfg)
    total_time = time.time() - start

    return path_cfg, total_time


def _solve_with_rrt(
    solver: RRTMotionPlanner,
    start_position: Array,
    start_wxyz: Array,
    target_position: Array,
    target_wxyz: Array,
    world_geom: CollGeom,
):
    """Specific solving logic for RRT."""
    # Warm-up run
    print("Warm-up run for RRT solver...")
    cfg_pair = solver._ik_solver_best(
        start_wxyz, start_position, target_wxyz, target_position, [world_geom]
    )
    path_cfg = solver.find_path(cfg_pair[0], cfg_pair[1], [world_geom])
    jax.block_until_ready(path_cfg)

    # Solve
    print(f"start solve rrt")
    start = time.time()
    cfg_pair = solver._ik_solver_best(
        start_wxyz, start_position, target_wxyz, target_position, [world_geom]
    )

    path_cfg = solver.find_path(cfg_pair[0], cfg_pair[1], [world_geom])

    jax.block_until_ready(path_cfg)
    total_time = time.time() - start

    return path_cfg, total_time


def _eval_planner(
    planner_name: str,
    solver,
    solve_fn: Callable,
    robot: CCRobot,
    batched_fk: Callable[[ConstantCurvatureState], Array],
    robot_coll: RobotCollision,
    world_geom: CollGeom,
    start_states: ConstantCurvatureState,
    end_states: ConstantCurvatureState,
    save_path: str,
):
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

    # Call the specific solver function
    path_cfg, total_time = solve_fn(
        solver, start_position, start_wxyz, target_position, target_wxyz, world_geom
    )

    # Process results
    is_valid = path_cfg is not None and path_cfg.theta.shape[0] > 0
    if is_valid:
        solution_states = jax.tree_util.tree_map(lambda x: x[-1:], path_cfg)
        all_paths = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), path_cfg
        )
        paths_are_valid = jnp.array([True])

    else:
        print(
            f"[Warning] No solution found by {method_name_upper}, using start state as placeholder."
        )
        return {
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

    fk_result = batched_fk(path_cfg)
    planned_tip_traj = jaxlie.SE3.from_matrix(fk_result[:, -1, ...])

    # Save results
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            start_states_theta=np.asarray(start_states.theta),
            start_states_phi=np.asarray(start_states.phi),
            end_states_theta=np.asarray(end_states.theta),
            end_states_phi=np.asarray(end_states.phi),
            target_position=np.asarray(target_position),
            target_wxyz=np.asarray(target_wxyz),
            fk_result=np.asarray(fk_result),
            solution_states_theta=np.asarray(solution_states.theta),
            solution_states_phi=np.asarray(solution_states.phi),
            planned_tip_traj=np.asarray(planned_tip_traj.as_matrix()),
        )
        print(f"Saved {method_name_upper} results data to {save_path}")

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
    print(f"--- {method_name_upper} With Collision Results ---")
    print(
        f"Kinematic Reachability Rate (accurate AND within limits): {kinematic_reachability_rate:.2f}%"
    )
    print(
        f"Final Success Rate (accurate, within limits, AND collision-free): {final_success_rate:.2f}%"
    )
    print(f"Final Position Error: {final_pos_error:.3f}m")
    print(f"Final Rotation Error: {final_rot_error:.3f}rad")
    print(
        f"finish solve {planner_name} of num sections {robot.config.num_sections}, total time: {total_time}s"
    )
    print(failure_stats)

    return {
        "method": method_name_upper,
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


def eval_mp_all_sections(
    robot_config_path: str,
    world_config_path: str,
    section_list: list,
    eval_num: int,
    save_dir: str,
    min_sample_dist_ratio: float,
    planner_type: str = "trajopt",
) -> list:
    all_results_summary = []

    planner_map = {
        "trajopt": (_solve_with_trajopt, MotionPlanner),
        "prm": (_solve_with_prm, SamplingBasedMotionPlanner),
        "rrt": (_solve_with_rrt, RRTMotionPlanner),
    }

    if planner_type not in planner_map:
        raise ValueError(
            f"Unknown planner_type: '{planner_type}'. Available options are {list(planner_map.keys())}"
        )

    solve_fn, solver_class = planner_map[planner_type]

    for num_sections in section_list:
        # load robot config
        config = json.load(open(robot_config_path))
        config["num_sections"] = num_sections

        robot = CCRobot.from_config(config)
        robot_coll = RobotCollision.from_config(config)
        robot_total_length = robot.config.length * robot.config.num_sections
        # load world config
        world_coll = WorldCollision.from_config(world_config_path)

        batched_fk = robot.forward_kinematics

        solver = solver_class(robot, robot_coll, timesteps=100)
        world_geom = world_coll.collision_geoms_no_ground[-1]

        # Sample all pairs first
        print(
            f"\n--- Sampling {eval_num} pairs for {planner_type.upper()} with {num_sections} sections ---"
        )
        sample_data_path = (
            f"{save_dir}/sampled_states/sections_{num_sections}_eval_{eval_num}.npz"
        )
        start_states, end_states = sample_collision_free_start_end_states(
            robot=robot,
            eval_num=eval_num,
            robot_coll=robot_coll,
            world_geom=world_geom,
            batched_fk=batched_fk,
            min_distance=robot_total_length * min_sample_dist_ratio,
            save_load_path=sample_data_path,
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
        for i in range(actual_eval_num):
            print(
                f"\n--- Evaluating Pair {i+1}/{actual_eval_num} for {planner_type.upper()} with {num_sections} sections ---"
            )

            # Select the i-th start and end state from the sampled pairs
            start_state_i = jax.tree_util.tree_map(
                lambda x: x[i : i + 1], start_states
            )
            end_state_i = jax.tree_util.tree_map(lambda x: x[i : i + 1], end_states)

            # Save the detailed trajectory of each trial
            save_path = f"{save_dir}/{planner_type}_sections_{num_sections}_trial_{i}.npz"

            result = _eval_planner(
                planner_name=planner_type,
                solver=solver,
                solve_fn=solve_fn,
                robot=robot,
                batched_fk=batched_fk,
                robot_coll=robot_coll,
                world_geom=world_geom,
                start_states=start_state_i,
                end_states=end_state_i,
                save_path=save_path,
            )
            trial_results.append(result)

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
        full_results_path = f"{save_dir}/all_trials_results/{planner_type}_sections_{num_sections}_all_trials_results.npz"
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
            f"\nSaved full results for {planner_type.upper()} with {num_sections} sections to {full_results_path}"
        )

        # Create the aggregated summary dictionary
        summary = {
            "method": planner_type.upper(),
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


if __name__ == "__main__":
    test_list = [3,4,5,6]
    repeat_num = 50  # Evaluate 10 times for each configuration
    robot_config_path = "configs/robots/cc_scene_eval.json"
    # world_config_path = "configs/maps/mp_maps/obstacles_lattice.json"
    world_config_path = "configs/maps/mp_scene/obstacles_13.pick_from_shelf.json"
    # result_dir = "results/2.pick_from_box"

    # TODO: change the result_dir to the scene name
    result_dir = "results/13.pick_from_shelf"
    os.makedirs(result_dir, exist_ok=True)
    planner_types = ["rrt"] # ["trajopt", "prm", "rrt"]
    result_summarys = []

    for planner_type in planner_types:
        result_summary = eval_mp_all_sections(
            robot_config_path=robot_config_path,
            world_config_path=world_config_path,
            section_list=test_list,
            eval_num=repeat_num,
            save_dir=result_dir,
            min_sample_dist_ratio=0.1,
            planner_type=planner_type,
        )
        result_summarys.extend(result_summary)

    summarize_results(result_summarys)

