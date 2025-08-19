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
from soul.robots.cc_robot_extend import (
    CCRobot as CCRobotExtend,
    ConstantCurvatureState as ConstantCurvatureStateExtend,
)
from soul.solver import IKSolver
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


def ik_metric_with_coll(
    robot: CCRobot,
    solution: ConstantCurvatureState,
    result_transform: jaxlie.SE3,
    target_position: Array,
    target_orientation: Array,
    solution_collision_mask: Array,
) -> tuple[float, float, dict]:
    """
    Calculates IK success metrics considering accuracy, joint limits, and collision avoidance.

    Args:
        robot: The CC robot model
        solution: The solution states from the IK solver
        result_transform: The resulting end-effector transforms from the IK solver.
        target_position: The target positions.
        target_orientation: The target orientations (as wxyz quaternions).
        solution_collision_mask: A boolean array where True indicates a collision.

    Returns:
        A tuple containing:
        - final_success_rate: The percentage of solutions that are accurate, within limits, and collision-free.
        - final_error: The mean error for successful solutions.
        - failure_stats: Detailed failure statistics dictionary.
    """
    target_transform = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_orientation), target_position
    )
    error = jnp.linalg.norm(
        (target_transform.inverse() @ result_transform).log(), axis=-1
    )

    # Individual failure masks for accuracy
    acc_mask = error < 0.01

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

    if isinstance(robot, CCRobotExtend):
        length_mask = jnp.all(
            jnp.logical_and(
                solution.length >= robot.config.lower_limits_length - delta,
                solution.length <= robot.config.upper_limits_length + delta,
            ),
            axis=-1,
        )
        length_fail_mask = ~length_mask
        joint_limits_mask = joint_limits_mask & length_mask
        num_length_fail = jnp.sum(length_fail_mask)

    # Combined masks
    accuracy_and_limits_mask = acc_mask & joint_limits_mask

    # Final success mask (accurate, within limits, and collision-free)
    final_success_mask = jnp.logical_and(
        accuracy_and_limits_mask,
        ~solution_collision_mask,
    )

    # Calculate detailed failure statistics
    total_samples = len(final_success_mask)
    num_success = jnp.sum(final_success_mask)
    num_fail = total_samples - num_success

    # Individual failure types
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
        # Percentages
        "theta_fail_rate": float(num_theta_fail / total_samples * 100),
        "phi_fail_rate": float(num_phi_fail / total_samples * 100),
        "length_fail_rate": (
            float(num_length_fail / total_samples * 100)
            if isinstance(robot, CCRobotExtend)
            else 0
        ),
        "collision_fail_rate": float(num_collision_fail / total_samples * 100),
        "accuracy_fail_rate": float(num_accuracy_fail / total_samples * 100),
        "limit_fail_rate": float(num_limit_fail / total_samples * 100),
    }

    final_success_rate = jnp.mean(final_success_mask) * 100.0
    # Use jnp.nanmean to avoid errors if no solutions are successful
    final_error = jnp.nan_to_num(jnp.mean(error[final_success_mask]))

    return final_success_rate, final_error, failure_stats


def sample_states_test(robot: CCRobot, num_states: int) -> ConstantCurvatureState:
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

    if isinstance(robot, CCRobotExtend):
        length = jax.random.uniform(
            key=subkey,
            shape=(num_states, robot.config.num_sections),
            minval=robot.config.lower_limits_length,
            maxval=robot.config.upper_limits_length,
        )

        states = ConstantCurvatureStateExtend(
            base_position=jnp.zeros((num_states, 3)),
            theta=theta,
            phi=phi,
            length=length,
        )
    else:
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
    world_dist = robot_coll.compute_world_collision_distance(robot, state, world_geom)
    return jnp.any(jnp.any(world_dist < 0) == True)


def save_targets_to_csv(target_wxyz, target_position, num_sections, eval_num):
    import csv
    import os
    
    # Create directory if it doesn't exist
    os.makedirs("results/ik_coll", exist_ok=True)
    
    # Save target positions and orientations to CSV
    csv_path = f"results/ik_coll/targets_{num_sections}_{eval_num}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['target_wx', 'target_wy', 'target_wz', 'target_ww', 
                    'target_px', 'target_py', 'target_pz']
        writer.writerow(header)
        
        # Write data - each row is one sample
        for i in range(eval_num):
            row = [
                float(target_wxyz[i, 0]),
                float(target_wxyz[i, 1]), 
                float(target_wxyz[i, 2]),
                float(target_wxyz[i, 3]),
                float(target_position[i, 0]),
                float(target_position[i, 1]),
                float(target_position[i, 2])
            ]
            writer.writerow(row)

def eval_ik_with_coll(
    robot: CCRobot,
    eval_num: int,
    batched_ik_solve: Callable[[Array, Array, Sequence[CollGeom]], Array],
    batched_fk: Callable[[ConstantCurvatureState], Array],
    robot_coll: RobotCollision,
    world_geom: CollGeom,
    save_path: str,
):
    """Main function for basic IK with collision avoidance."""
    num_sections = robot.config.num_sections
    print(
        f"start solve ik WITH COLLISION of num sections {num_sections}, num eval {eval_num}"
    )

    # Sample collision-free target points
    print(f"Sampling {eval_num} collision-free target states...")
    collision_free_states = []
    total_sampled = 0
    max_sampling_attempts = eval_num * 10000  # Safety break for the while loop
    num_free_states = 0

    while num_free_states < eval_num and total_sampled < max_sampling_attempts:
        # Batch sample a set of states
        candidate_states = sample_states_test(robot, eval_num)
        total_sampled += eval_num

        # JAX-style check collision for a batch of states
        is_collision_vmap = jax.vmap(
            is_state_in_collision, in_axes=(0, None, None, None)
        )
        collision_results = is_collision_vmap(
            candidate_states, robot, robot_coll, world_geom
        )
        collision_free_states.append(candidate_states[~collision_results])
        num_free_states += jnp.sum(~collision_results)

    # Merge the list of collision-free states into a single Pytree
    initial_states = jax.tree_util.tree_map(
        lambda *x: jnp.concatenate(x), *collision_free_states
    )
    print(f"Finish sampling. Total nums of free states checked: {num_free_states}")

    # Generate target poses, visualization, warmup similar to before
    target_transforms = batched_fk(initial_states[:eval_num])
    tip_transform = jaxlie.SE3.from_matrix(target_transforms[:, -1, ...])
    target_wxyz = tip_transform.rotation().wxyz
    target_position = tip_transform.translation()
    save_targets_to_csv(target_wxyz, target_position, num_sections, eval_num)
    # warmup
    print(f"finish forward, start warmup")
    jax.block_until_ready(batched_ik_solve(target_wxyz, target_position, [world_geom]))

    # solve ik
    start = time.time()
    print("start solve ik")
    solution_states = batched_ik_solve(target_wxyz, target_position, [world_geom])
    jax.block_until_ready(solution_states)
    total_time = time.time() - start

    # calculate accuracy metrics
    fk_result = robot.forward_kinematics(solution_states)
    tip_transforms = jaxlie.SE3.from_matrix(fk_result[:, -1, ...])

    # Save target and fk_result data
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if isinstance(robot, CCRobotExtend):
            np.savez(
                save_path,
                sampled_theta=np.array(initial_states.theta),
                sampled_phi=np.array(initial_states.phi),
                sampled_length=np.array(initial_states.length),
                target_position=np.array(target_position),
                target_wxyz=np.array(target_wxyz),
                fk_result=np.array(fk_result),
                solution_states_theta=np.array(solution_states.theta),
                solution_states_phi=np.array(solution_states.phi),
                solution_states_length=np.array(solution_states.length),
            )
        else:
            np.savez(
                save_path,
                sampled_theta=np.array(initial_states.theta),
                sampled_phi=np.array(initial_states.phi),
                target_position=np.array(target_position),
                target_wxyz=np.array(target_wxyz),
                fk_result=np.array(fk_result),
                solution_states_theta=np.array(solution_states.theta),
                solution_states_phi=np.array(solution_states.phi),
            )
        print(f"Saved target and fk_result data to {save_path}")

    # metrics
    is_solution_collision_vmap = jax.vmap(
        is_state_in_collision, in_axes=(0, None, None, None)
    )
    solution_collision_mask = is_solution_collision_vmap(
        solution_states, robot, robot_coll, world_geom
    )
    final_success_rate, final_error, failure_stats = ik_metric_with_coll(
        robot,
        solution_states,
        tip_transforms,
        target_position,
        target_wxyz,
        solution_collision_mask,
    )

    print(f"--- With Collision Results ---")
    print(
        f"Final Success Rate (accurate, within limits, AND collision-free): {final_success_rate:.2f}%"
    )
    print(f"finish solve ik of num sections {num_sections}, total time: {total_time}s")

    # Print detailed failure analysis
    print("\n--- Detailed Failure Analysis ---")
    print(f"Total samples: {failure_stats['total_samples']}")
    print(
        f"Successful: {failure_stats['num_success']} ({failure_stats['success_rate']:.2f}%)"
    )
    print(
        f"Failed: {failure_stats['num_fail']} ({100 - failure_stats['success_rate']:.2f}%)"
    )
    print(failure_stats)

    return {
        "eval num": eval_num,
        "num sections": num_sections,
        "error": final_error,
        "success rate": final_success_rate,
        "total time": total_time,
        "failure_stats": failure_stats,
    }


def eval_ik_all_sections(
    robot_config_path: str,
    world_config_path: str,
    section_list: list,
    eval_num_list: list,
    eval_type: str,
    save_dir: str,
):
    all_results_summary = []
    for num_sections in section_list:
        # load robot config
        config = json.load(open(robot_config_path))
        config["num_sections"] = num_sections
        if eval_type == "cc":
            robot = CCRobot.from_config(config)
        elif eval_type == "cc_extend":
            robot = CCRobotExtend.from_config(config)
        else:
            raise ValueError(f"Invalid eval type: {eval_type}")
        robot_coll = RobotCollision.from_config(config)

        # load world config
        world_coll = WorldCollision.from_config(world_config_path)

        batched_fk = jax.vmap(robot._forward_kinematics)
        solver = IKSolver(
            robot,
            num_seeds_init=128,
            num_seeds_final=8,
            total_steps=100,
            init_steps=10,
            coll=robot_coll,
        )
        batched_ik_solve = jax.vmap(
            jax.jit(solver.solve_ik_best_with_coll), in_axes=(0, 0, None)
        )
        for eval_num in eval_num_list:
            # Create save path for this specific evaluation
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            save_path = (
                f"{save_dir}/ik_with_coll_sections_{num_sections}_eval_{eval_num}.npz"
            )
            all_results_summary.append(
                eval_ik_with_coll(
                    robot,
                    eval_num,
                    batched_ik_solve,
                    batched_fk,
                    robot_coll,
                    world_coll.collision_geoms_no_ground[-1],
                    save_path,
                )
            )

    print("\n\n--- IK test resume ---")
    header = f"{'num sections':<10} | {'eval num':<15} | {'error':<15} | {'success rate (%)':<15} | {'total time (s)':<18} "
    print(header)
    print("-" * len(header))
    for res_item in all_results_summary:
        eval_num_str = f"{res_item['eval num']}"
        error_str = f"{res_item['error']}"
        sr_str = f"{res_item['success rate']:.2f}"
        time_str = f"{res_item['total time']:.3f}"
        print(
            f"{res_item['num sections']:<10} | {eval_num_str:<15} | {error_str:<15} | {sr_str:<15} | {time_str:<18}"
        )


if __name__ == "__main__":
    test_list = [6]
    eval_num_list = [100]
    robot_config_path = "configs/robots/cc_extend_eval.json"
    # robot_config_path = "configs/robots/cc_eval.json"
    # world_config_path = "configs/maps/ik_maps/obstacles_lattice.json"
    world_config_path = "configs/maps/ik_maps/obstacles_icosahedron.json"
    result_dir = "results/ik_with_coll_cube"
    eval_ik_all_sections(
        robot_config_path,
        world_config_path,
        test_list,
        eval_num_list,
        "cc_extend",
        result_dir,
    )
