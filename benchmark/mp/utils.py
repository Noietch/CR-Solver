import jax
from jaxtyping import Array
import numpy as np
import jax.numpy as jnp
import jaxlie
import os
from typing import Callable, Sequence
from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.geom import (
    RobotCollision,
    CollGeom,
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
    delta = float("inf")  # the margin for the joint limit constraint
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
    world_geom: Sequence[CollGeom],
    batched_fk: Callable[[ConstantCurvatureState], Array],
    min_distance: float = 0.1,
    batch_size: int = 256,
    save_load_path: str = None,
    start_from_initialization: bool = False,
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
        print(f"Loaded {start_states.theta.shape[0]} start/end pairs.")
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
    used_indices = jnp.zeros(state_pool.theta.shape[0], dtype=bool)
    num_available = state_pool.theta.shape[0]
    max_pairing_attempts = num_available * 10  # Safety break for pairing

    print("Pairing states...")

    if start_from_initialization:
        # initialization with theta, phi = 0
        print("Adding initialization states (theta=0, phi=0) as start states...")

        # Create single initialization state with theta=0, phi=0
        init_state = ConstantCurvatureState(
            base_position=jnp.zeros((1, 3)),
            theta=jnp.full((1, robot.config.num_sections), 1e-7),
            phi=jnp.full((1, robot.config.num_sections), 1e-7),
        )

        for _ in range(max_pairing_attempts):
            if len(final_start_states_list) >= eval_num:
                break
            if np.sum(~used_indices) < 2:
                print("Warning: Not enough available states to form a new pair.")
                break

            # Get available indices
            available_indices = jnp.where(~used_indices)[0]
            # Pick two random, unique indices from the available pool
            idx2 = np.random.choice(np.array(available_indices), size=1, replace=False)
            idx2_int = int(idx2.item())

            init_fk_transforms = batched_fk(init_state)
            init_tip_transform = jaxlie.SE3.from_matrix(init_fk_transforms[0, -1, ...])

            pos1 = init_tip_transform.translation()
            pos2 = positions_pool[idx2_int]

            distance = jnp.linalg.norm(pos2 - pos1)

            if distance >= min_distance:
                start_state = init_state
                end_state = jax.tree_util.tree_map(
                    lambda x: x[idx2_int : idx2_int + 1], state_pool
                )

                final_start_states_list.append(start_state)
                final_end_states_list.append(end_state)

                used_indices = used_indices.at[idx2_int].set(True)

                if (
                    len(final_start_states_list) % 10 == 0
                    or len(final_start_states_list) == eval_num
                ):
                    print(
                        f"  ... found {len(final_start_states_list)}/{eval_num} pairs"
                    )

    else:
        for _ in range(max_pairing_attempts):
            if len(final_start_states_list) >= eval_num:
                break
            if np.sum(~used_indices) < 2:
                print("Warning: Not enough available states to form a new pair.")
                break

            # Get available indices
            available_indices = jnp.where(~used_indices)[0]
            # Pick two random, unique indices from the available pool
            idx1, idx2 = np.random.choice(
                np.array(available_indices), size=2, replace=False
            )

            pos1 = positions_pool[idx1]
            pos2 = positions_pool[idx2]

            distance = jnp.linalg.norm(pos1 - pos2)

            if distance >= min_distance:
                start_state = jax.tree_util.tree_map(
                    lambda x: x[idx1 : idx1 + 1], state_pool
                )
                end_state = jax.tree_util.tree_map(
                    lambda x: x[idx2 : idx2 + 1], state_pool
                )

                final_start_states_list.append(start_state)
                final_end_states_list.append(end_state)

                used_indices = used_indices.at[idx1].set(True)
                used_indices = used_indices.at[idx2].set(True)

                if (
                    len(final_start_states_list) % 10 == 0
                    or len(final_start_states_list) == eval_num
                ):
                    print(
                        f"  ... found {len(final_start_states_list)}/{eval_num} pairs"
                    )

    if len(final_start_states_list) < eval_num:
        print(
            f"Warning: Could only sample {len(final_start_states_list)} pairs "
            f"out of {eval_num} required."
        )
        if not final_start_states_list:
            raise RuntimeError("Could not sample any valid state pairs.")

    final_start_states: ConstantCurvatureState = jax.tree_util.tree_map(
        lambda *x: jnp.concatenate(x, axis=0), *final_start_states_list
    )
    final_end_states: ConstantCurvatureState = jax.tree_util.tree_map(
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


def delete_failed_states(
    save_load_path: str,
    failed_indices: Array,
    backup_suffix: str = "_backup"
) -> tuple[ConstantCurvatureState, ConstantCurvatureState]:
    """
    Load start/end states from file and remove failed trials based on their indices.
    
    Args:
        save_load_path: Path to the saved states file
        failed_indices: Array of indices corresponding to failed trials
        backup_suffix: Suffix to add to the original file for backup
        
    Returns:
        Tuple of (filtered_start_states, filtered_end_states) with failed trials removed
    """
    if not (save_load_path and os.path.exists(save_load_path)):
        raise FileNotFoundError(f"States file not found: {save_load_path}")
        
    print(f"Loading pre-sampled states from {save_load_path}...")
    data = np.load(save_load_path)
    
    # Load original states
    original_start_theta = data["start_theta"]
    original_start_phi = data["start_phi"] 
    original_end_theta = data["end_theta"]
    original_end_phi = data["end_phi"]
    
    total_trials = original_start_theta.shape[0]
    print(f"Loaded {total_trials} start/end pairs.")
    
    if len(failed_indices) > 0:
        print(f"Removing {len(failed_indices)} failed trials...")
        
        # Create backup of original file
        backup_path = save_load_path.replace('.npz', f'{backup_suffix}.npz')
        if not os.path.exists(backup_path):
            print(f"Creating backup at {backup_path}")
            np.savez(backup_path, 
                    start_theta=original_start_theta,
                    start_phi=original_start_phi, 
                    end_theta=original_end_theta,
                    end_phi=original_end_phi)
        
        # Create mask for successful trials (keep all indices except failed ones)
        all_indices = jnp.arange(total_trials)
        success_mask = jnp.isin(all_indices, failed_indices, invert=True)
        
        # Filter out failed trials
        filtered_start_theta = original_start_theta[success_mask]
        filtered_start_phi = original_start_phi[success_mask]
        filtered_end_theta = original_end_theta[success_mask]
        filtered_end_phi = original_end_phi[success_mask]
        
        # Save filtered data back to original file
        print(f"Saving {filtered_start_theta.shape[0]} successful trials back to {save_load_path}")
        np.savez(save_load_path,
                start_theta=filtered_start_theta,
                start_phi=filtered_start_phi,
                end_theta=filtered_end_theta, 
                end_phi=filtered_end_phi)
                
        # Create ConstantCurvatureState objects for successful trials
        start_states = ConstantCurvatureState(
            theta=jnp.array(filtered_start_theta),
            phi=jnp.array(filtered_start_phi),
            base_position=jnp.zeros((filtered_start_theta.shape[0], 3)),
        )
        end_states = ConstantCurvatureState(
            theta=jnp.array(filtered_end_theta),
            phi=jnp.array(filtered_end_phi),
            base_position=jnp.zeros((filtered_end_theta.shape[0], 3)),
        )
        
        print(f"Successfully removed failed trials. Remaining: {start_states.theta.shape[0]} pairs.")
        
    else:
        print("No failed indices provided, returning original states.")
        # Create ConstantCurvatureState objects for all original states
        start_states = ConstantCurvatureState(
            theta=jnp.array(original_start_theta),
            phi=jnp.array(original_start_phi),
            base_position=jnp.zeros((original_start_theta.shape[0], 3)),
        )
        end_states = ConstantCurvatureState(
            theta=jnp.array(original_end_theta),
            phi=jnp.array(original_end_phi),
            base_position=jnp.zeros((original_end_theta.shape[0], 3)),
        )
    
    return start_states, end_states


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


# @jax.jit
def is_state_in_self_collision(
    state: ConstantCurvatureState,
    robot: CCRobot,
    robot_coll: RobotCollision,
) -> bool:
    """
    Check if the robot is in self-collision.
    """
    self_collision_distances = robot_coll.compute_self_collision_distance(robot, state)
    # breakpoint()

    return jnp.any(self_collision_distances < 0.0)


# @jax.jit
def is_state_in_collision(
    state: ConstantCurvatureState,
    robot: CCRobot,
    robot_coll: RobotCollision,
    world_geom: Sequence[CollGeom],
) -> bool:
    """
    Check if the robot is in collision with obstacles or itself.

    A collision is defined as any distance less than 0 (i.e., penetration)
    between the robot and any world geometry, or between parts of the robot itself.
    """

    # Check for collision with each world geometry object.
    def check_single_geom(geom: CollGeom) -> bool:
        # compute_world_collision_distance returns signed distances.
        # A negative value indicates penetration/collision.
        world_dist = robot_coll.compute_world_collision_distance(robot, state, geom)
        return jnp.any(world_dist < 0.0)

    # Create a boolean array indicating if a collision occurs with each geometry.
    collision_results = jnp.array([check_single_geom(g) for g in world_geom])

    # # Check for self-collision
    # self_collision = is_state_in_self_collision(state, robot, robot_coll)

    # # If any of the checks returned True, there is a collision.
    # print(f"Collision check: {jnp.logical_or(jnp.any(collision_results), self_collision)}")
    # jnp.logical_or(jnp.any(collision_results), self_collision)
    # breakpoint()
    return jnp.any(collision_results)


@jax.jit
def is_path_in_collision(
    start_state: ConstantCurvatureState,
    end_state: ConstantCurvatureState,
    robot: CCRobot,
    robot_coll: RobotCollision,
    world_geom: Sequence[CollGeom],
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
    world_geom: Sequence[CollGeom],
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
        f"{'Method':<10} | {'Optimized':<10} | {'Sections':<10} | {'Eval Num':<10} | "
        f"{'Success (%)':<18} | {'Reachable (%)':<20} | "
        f"{'Pos Error (m)':<20} | {'Rot Error (rad)':<20} | {'Time (s)':<20}"
    )
    print(header)
    print("-" * len(header))

    for res_item in all_results_summary:
        method_str = res_item.get("method", "MP")
        optimized_str = str(res_item.get("with_optimization", "N/A"))
        sections_str = str(res_item.get("num sections", "N/A"))
        eval_num_str = str(res_item.get("eval num", "N/A"))

        # Format statistics with mean ± std
        sr_str = (
            f"{res_item['success_rate_mean']:.2f} ± {res_item['success_rate_std']:.2f}"
        )
        kr_str = f"{res_item['kinematic_rate_mean']:.2f} ± {res_item['kinematic_rate_std']:.2f}"
        ps_error_str = (
            f"{res_item['pos_error_mean']:.2f} ± {res_item['pos_error_std']:.2f}"
        )
        rt_error_str = (
            f"{res_item['rot_error_mean']:.2f} ± {res_item['rot_error_std']:.2f}"
        )
        time_str = f"{res_item['time_mean']:.3f} ± {res_item['time_std']:.3f}"

        print(
            f"{method_str:<10} | {optimized_str:<10} | {sections_str:<10} | {eval_num_str:<10} | "
            f"{sr_str:<18} | {kr_str:<20} | "
            f"{ps_error_str:<20} | {rt_error_str:<20} | {time_str:<20}"
        )
