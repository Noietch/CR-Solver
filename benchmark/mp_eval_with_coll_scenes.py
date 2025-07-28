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
## 改命名
## 改距离计算 done
## 调用motion_planner done
## 只考虑不可变长 done
## 清除无用代码（！去掉extend的东西） done
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
) -> tuple[ConstantCurvatureState, ConstantCurvatureState]:
    """
    Samples a specified number of pairs of collision-free start and end states,
    ensuring their end-effector positions are separated by a minimum distance.
    """
    print(
        f"Sampling {eval_num} collision-free start-end pairs with min distance {min_distance}..."
    )

    is_collision_vmap = jax.vmap(is_state_in_collision, in_axes=(0, None, None, None))

    # 1. Sample a large pool of collision-free states first.
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

    # 2. Pre-compute FK for the entire pool
    fk_transforms_pool = batched_fk(state_pool)
    tip_transforms_pool = jaxlie.SE3.from_matrix(fk_transforms_pool[:, -1, ...])
    positions_pool = tip_transforms_pool.translation()

    # 3. Pair states from the pool
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
    print("\n\n--- MP test resume ---")
    header = f"{'Method':<10} | {'num sections':<15} | {'eval num':<10} | {'Reachable (%)':<15} | {'Success (%)':<15} | {'Pos Error':<15} | {'Rot Error':<15} | {'Time (s)':<18} "
    print(header)
    print("-" * len(header))
    for res_item in all_results_summary:
        method_str = res_item.get("method", "MP")
        eval_num_str = f"{res_item['eval num']}"
        kr_str = f"{res_item['kinematic_reachability_rate']:.2f}"
        ps_error_str = f"{res_item['position error']:.4f}"
        rt_error_str = f"{res_item['rotation error']:.4f}"
        sr_str = f"{res_item['success rate']:.2f}"
        time_str = f"{res_item['total time']:.3f}"
        print(
            f"{method_str:<10} | {res_item['num sections']:<15} | {eval_num_str:<10} | {kr_str:<15} | {sr_str:<15} | {ps_error_str:<15} | {rt_error_str:<15} | {time_str:<18}"
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

    # 1. Get start and end poses from the states via FK
    start_transforms = batched_fk(start_states)
    start_tip_transform = jaxlie.SE3.from_matrix(start_transforms[0, -1, ...])
    start_wxyz = start_tip_transform.rotation().wxyz
    start_position = start_tip_transform.translation()

    target_transforms = batched_fk(end_states)
    target_tip_transform = jaxlie.SE3.from_matrix(target_transforms[0, -1, ...])
    target_wxyz = target_tip_transform.rotation().wxyz
    target_position = target_tip_transform.translation()

    # 2. Call the specific solver function
    path_cfg, total_time = solve_fn(
        solver, start_position, start_wxyz, target_position, target_wxyz, world_geom
    )

    # 3. Process results
    is_valid = path_cfg is not None and path_cfg.theta.shape[0] > 0
    if is_valid:
        solution_states = jax.tree_util.tree_map(lambda x: x[-1:], path_cfg)
        all_paths = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), path_cfg
        )
        paths_are_valid = jnp.array([True])
    else:
        print(
            f"No solution found by {method_name_upper}, using start state as placeholder."
        )
        solution_states = start_states
        # Return dummy values to match the expected structure
        all_paths = jax.tree_util.tree_map(
            lambda x: jnp.empty((0, solver.timesteps) + x.shape[1:], dtype=x.dtype),
            start_states,
        )
        paths_are_valid = jnp.array([False])

    # 4. Calculate accuracy metrics
    fk_result = batched_fk(solution_states)
    tip_transforms = jaxlie.SE3.from_matrix(fk_result[:, -1, ...])

    # 5. Save results
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            start_states_theta=np.array(start_states.theta),
            start_states_phi=np.array(start_states.phi),
            end_states_theta=np.array(end_states.theta),
            end_states_phi=np.array(end_states.phi),
            start_fk_result=np.array(start_transforms),
            target_position=np.array(target_position),
            target_wxyz=np.array(target_wxyz),
            target_fk_result=np.array(target_transforms),
            fk_result=np.array(fk_result),
            solution_states_theta=np.array(solution_states.theta),
            solution_states_phi=np.array(solution_states.phi),
            planned_tip_traj=np.array(tip_transforms.as_matrix()),
        )
        print(f"Saved {method_name_upper} results data to {save_path}")

    # 6. Calculate collision mask
    vmapped_is_trajectory_in_collision = jax.vmap(
        is_trajectory_in_collision, in_axes=(0, None, None, None)
    )
    path_collision_results = vmapped_is_trajectory_in_collision(
        all_paths, robot, robot_coll, world_geom
    )
    solution_collision_mask = jnp.logical_or(path_collision_results, ~paths_are_valid)

    # 7. Calculate final metrics
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

    # 8. Print and return results
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
    eval_num_list: list,
    save_dir: str,
    min_sample_dist_ratio: float,
    planner_type: str = "trajopt",
):
    all_results_summary = []

    planner_map = {
        "trajopt": (_solve_with_trajopt, MotionPlanner),
        "prm": (_solve_with_prm, SamplingBasedMotionPlanner),
        "rrt": (_solve_with_rrt, RRTMotionPlanner),
    }
    if planner_type == "mp":  # For backward compatibility
        planner_type = "trajopt"

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
        robot_length = robot.config.length * robot.config.num_sections
        # load world config
        world_coll = WorldCollision.from_config(world_config_path)

        batched_fk = jax.vmap(robot._forward_kinematics)

        solver = solver_class(robot, robot_coll, timesteps=100)
        world_geom = world_coll.collision_geoms_no_ground[-1]

        for i, eval_num in enumerate(eval_num_list):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            print(
                f"\n--- Starting Eval {i+1}/{len(eval_num_list)} for {planner_type.upper()} with {num_sections} sections ---"
            )

            # 1. Sample collision-free start and end points
            start_states, end_states = sample_collision_free_start_end_states(
                robot,
                1,
                robot_coll,
                world_geom,
                batched_fk,
                robot_length * min_sample_dist_ratio,
            )

            if start_states.theta.shape[0] == 0:
                print("Failed to sample a valid start/end pair. Skipping evaluation.")
                continue

            save_path = f"{save_dir}/{planner_type}_with_coll_sections_{num_sections}_eval_{i}.npz"

            result = _eval_planner(
                planner_name=planner_type,
                solver=solver,
                solve_fn=solve_fn,
                robot=robot,
                batched_fk=batched_fk,
                robot_coll=robot_coll,
                world_geom=world_geom,
                start_states=start_states,
                end_states=end_states,
                save_path=save_path,
            )
            all_results_summary.append(result)

    summarize_results(all_results_summary)


if __name__ == "__main__":
    test_list = [3, 4, 5, 6]
    eval_num_list = [1]  # Evaluate 5 times for each configuration
    robot_config_path = "configs/robots/cc_scene_eval.json"
    # world_config_path = "configs/maps/mp_maps/obstacles_lattice.json"
    world_config_path = "configs/maps/mp_scene/obstacles_test.json"
    # result_dir = "results/2.pick_from_box"
    result_dir = "results/test"
    eval_mp_all_sections(
        robot_config_path,
        world_config_path,
        test_list,
        eval_num_list,
        result_dir,
        min_sample_dist_ratio=0.1,
        planner_type="rrt",  # trajopt / prm / rrt
    )
