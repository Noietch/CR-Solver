import jax
from jaxtyping import Array
import numpy as np
import jax.numpy as jnp
import jaxlie
import os
from typing import Sequence, Any, Tuple, Union, List
import json
from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.robots.tdcr_robot import TDCRRobot
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
    position_threshold: float = 0.08
    rotation_threshold: float = 1.0

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


@jax.jit
def is_state_in_self_collision(
    state: ConstantCurvatureState,
    robot: CCRobot,
    robot_coll: RobotCollision,
) -> bool:
    """
    Check if the robot is in self-collision.
    """
    self_collision_distances = robot_coll.compute_self_collision_distance(robot, state)
    return jnp.any(self_collision_distances < 0.0)


@jax.jit
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

    def check_single_geom(geom: CollGeom) -> bool:
        world_dist = robot_coll.compute_world_collision_distance(robot, state, geom)
        return jnp.any(world_dist < 0.0)

    collision_results = jnp.array([check_single_geom(g) for g in world_geom])

    self_collision = is_state_in_self_collision(state, robot, robot_coll)

    return jnp.logical_or(jnp.any(collision_results), self_collision)


def log_result(
    base_info: tuple[
        ConstantCurvatureState,
        ConstantCurvatureState,
        TDCRRobot,
        RobotCollision,
        Sequence[CollGeom],
        callable,
        int,
    ],
    path_cfg: ConstantCurvatureState,
    method_name: str,
    trajopt_time: float = 0.0,
    prm_time: float = 0.0,
    opt_time: float = 0.0,
    rrt_time: float = 0.0,
    with_opt: bool = None,
):
    start_state, end_state, robot, robot_coll, world_geom, batched_fk, step = base_info
    start_state = jax.tree_util.tree_map(
        lambda x: jnp.expand_dims(x, axis=0), start_state
    )
    end_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), end_state)
    is_valid = path_cfg is not None and path_cfg.theta.shape[0] > 0
    total_time = trajopt_time + prm_time + opt_time + rrt_time

    # This dictionary will hold the data to be saved later.
    data_to_save = {}

    if is_valid:
        paths_are_valid = jnp.array([True])
        solution_states: ConstantCurvatureState = jax.tree_util.tree_map(
            lambda x: x[-1:], path_cfg
        )
        fk_result = batched_fk(path_cfg)
        planned_tip_traj = jaxlie.SE3.from_matrix(fk_result[:, -1, ...])

        # Calculate trajectory length
        tip_positions = planned_tip_traj.translation()  # shape: (T, 3)
        traj_segment_lengths = jnp.linalg.norm(
            tip_positions[1:] - tip_positions[:-1], axis=-1
        )
        traj_length = jnp.sum(traj_segment_lengths)

        start_transforms = batched_fk(start_state)
        start_tip_transform = jaxlie.SE3.from_matrix(start_transforms[0, -1, ...])
        start_wxyz = start_tip_transform.rotation().wxyz
        start_position = start_tip_transform.translation()
        tip_transforms = jax.tree_util.tree_map(lambda x: x[-1:], planned_tip_traj)

        target_transforms = batched_fk(end_state)
        target_tip_transform = jaxlie.SE3.from_matrix(target_transforms[0, -1, ...])
        target_wxyz = target_tip_transform.rotation().wxyz
        target_position = target_tip_transform.translation()

        vmapped_is_trajectory_in_collision = jax.vmap(
            is_trajectory_in_collision, in_axes=(0, None, None, None)
        )
        all_paths = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), path_cfg
        )
        path_collision_results = vmapped_is_trajectory_in_collision(
            all_paths, robot, robot_coll, world_geom
        )
        solution_collision_mask = jnp.logical_or(
            path_collision_results, ~paths_are_valid
        )
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

        # Prepare data for saving
        data_to_save = {
            "trial_id": step,
            "method": method_name,
            "with_opt": with_opt,
            "total_time": total_time,
            "trajopt_time": trajopt_time,
            "prm_time": prm_time,
            "opt_time": opt_time,
            "rrt_time": rrt_time,
            "start_position": np.asarray(start_position),
            "start_wxyz": np.asarray(start_wxyz),
            "target_position": np.asarray(target_position),
            "target_wxyz": np.asarray(target_wxyz),
            "start_states_theta": np.asarray(start_state.theta),
            "start_states_phi": np.asarray(start_state.phi),
            "end_states_theta": np.asarray(end_state.theta),
            "end_states_phi": np.asarray(end_state.phi),
            "fk_result": np.asarray(fk_result),
            "solution_states_theta": np.asarray(solution_states.theta),
            "solution_states_phi": np.asarray(solution_states.phi),
            "traj_length": float(traj_length),
            "planned_paths": path_cfg.save_dict(),
            "planned_tip_traj": np.asarray(planned_tip_traj.as_matrix()),
            "is_success": final_success_rate > 99.0,
            "kinematic_reachability": kinematic_reachability_rate > 99.0,
            "final_pos_error": final_pos_error,
            "final_rot_error": final_rot_error,
            "failure_stats": failure_stats,
        }
        successful_id = step if final_success_rate > 99.0 else None
        print(
            f"[{step}] [{method_name}({'optimized' if with_opt else 'no opt'})] found a solution in {total_time:.3f}s \n"
            f"[{'Success' if final_success_rate > 99.0 else 'Failure'}], \n"
            f"Kinematic Reachability: {kinematic_reachability_rate:.0f}%, \n"
            f"Pos Error: {final_pos_error:.3f}m, Rot Error: {final_rot_error:.3f}rad \n"
        )
    else:
        print(f"[Warning] No solution found by {method_name}")
        # Return summary and empty data dictionary
        successful_id = None
        return data_to_save, successful_id, None, None

    return data_to_save, successful_id, total_time, float(traj_length)


def remove_none(*seqs: Sequence[Any]) -> Union[List[Any], Tuple[List[Any], ...]]:
    filtered = [[x for x in seq if x is not None] for seq in seqs]
    if len(filtered) == 1:
        return filtered[0]
    return tuple(filtered)


def save_result(
    save_dir: str,
    world_config_path: str,
    eval_num: int,
    actual_eval_num: int,
    num_sections: int,
    road_map_nodes: int,
    trajopt_success: List[int],
    traj_opt_total_time: List[float],
    traj_opt_traj_length: List[float],
    prm_success: List[int],
    prm_total_time: List[float],
    prm_traj_length: List[float],
    prm_opt_success: List[int],
    prm_opt_total_time: List[float],
    prm_opt_traj_length: List[float],
    rrt_success: List[int],
    rrt_total_time: List[float],
    rrt_traj_length: List[float],
    rrt_opt_success: List[int],
    rrt_opt_total_time: List[float],
    rrt_opt_traj_length: List[float],
):
    (
        trajopt_success,
        traj_opt_total_time,
        traj_opt_traj_length,
        prm_success,
        prm_total_time,
        prm_traj_length,
        prm_opt_success,
        prm_opt_total_time,
        prm_opt_traj_length,
        rrt_success,
        rrt_total_time,
        rrt_traj_length,
        rrt_opt_success,
        rrt_opt_total_time,
        rrt_opt_traj_length,
    ) = remove_none(
        trajopt_success,
        traj_opt_total_time,
        traj_opt_traj_length,
        prm_success,
        prm_total_time,
        prm_traj_length,
        prm_opt_success,
        prm_opt_total_time,
        prm_opt_traj_length,
        rrt_success,
        rrt_total_time,
        rrt_traj_length,
        rrt_opt_success,
        rrt_opt_total_time,
        rrt_opt_traj_length,
    )

    trajopt_success_rate = len(trajopt_success) / actual_eval_num
    prm_success_rate = len(prm_success) / actual_eval_num
    prm_opt_success_rate = len(prm_opt_success) / actual_eval_num
    rrt_success_rate = len(rrt_success) / actual_eval_num
    rrt_opt_success_rate = len(rrt_opt_success) / actual_eval_num

    traj_time_avg = float(np.array(traj_opt_total_time).mean())
    prm_time_avg = float(np.array(prm_total_time).mean())
    prm_opt_time_avg = float(np.array(prm_opt_total_time).mean())
    rrt_time_avg = float(np.array(rrt_total_time).mean())
    rrt_opt_time_avg = float(np.array(rrt_opt_total_time).mean())

    traj_opt_traj_length = float(np.array(traj_opt_traj_length).mean())
    prm_traj_length = float(np.array(prm_traj_length).mean())
    prm_opt_traj_length = float(np.array(prm_opt_traj_length).mean())
    rrt_traj_length = float(np.array(rrt_traj_length).mean())
    rrt_opt_traj_length = float(np.array(rrt_opt_traj_length).mean())

    log_data = {
        "scene_name": os.path.splitext(os.path.basename(world_config_path))[0],
        "num_sections": num_sections,
        "eval_num": actual_eval_num,
        "prm_road_map_nodes": road_map_nodes,
        "trajopt_success_rate": trajopt_success_rate,
        "prm_success_rate": prm_success_rate,
        "prm_opt_success_rate": prm_opt_success_rate,
        "rrt_success_rate": rrt_success_rate,
        "rrt_opt_success_rate": rrt_opt_success_rate,
        "traj_time_avg": traj_time_avg,
        "prm_time_avg": prm_time_avg,
        "prm_opt_time_avg": prm_opt_time_avg,
        "rrt_time_avg": rrt_time_avg,
        "rrt_opt_time_avg": rrt_opt_time_avg,
        "traj_opt_traj_length": traj_opt_traj_length,
        "prm_traj_length": prm_traj_length,
        "prm_opt_traj_length": prm_opt_traj_length,
        "rrt_traj_length": rrt_traj_length,
        "rrt_opt_traj_length": rrt_opt_traj_length,
        "traj_opt_success_id": trajopt_success,
        "prm_success_id": prm_success,
        "prm_opt_success_id": prm_opt_success,
        "rrt_success_id": rrt_success,
        "rrt_opt_success_id": rrt_opt_success,
    }
    print(log_data)
    results_json_path = os.path.join(
        save_dir, f"sections_{num_sections}_eval_{eval_num}_results.json"
    )
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"Saved results to {results_json_path}")
