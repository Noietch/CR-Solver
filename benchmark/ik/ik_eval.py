import jax
from jaxtyping import Array
import jax.numpy as jnp
import jaxlie
import time
import json
import os
from typing import Callable
from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.robots.cc_robot_extend import (
    CCRobot as CCRobotExtend,
    ConstantCurvatureState as ConstantCurvatureStateExtend,
)
from soul.solver import IKSolver

DISABLE_JIT = False

if DISABLE_JIT:
    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def ik_metric(
    robot: CCRobot,
    solution: ConstantCurvatureState,
    result_transform: jaxlie.SE3,
    target_position: Array,
    target_orientation: Array,
) -> tuple[float, float, float, dict]:
    # use the target position and orientation to compute the error
    target_transform = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_orientation), target_position
    )
    error = jnp.linalg.norm(
        (target_transform.inverse() @ result_transform).log(), axis=-1
    )

    # Individual failure masks
    acc_mask = error < 0.01

    # check the limit constraint
    delta = 0.01
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

    # Overall success mask
    mask = acc_mask & theta_mask & phi_mask

    # Calculate detailed failure statistics
    total_samples = len(mask)
    num_success = jnp.sum(mask)
    num_fail = total_samples - num_success

    # Detailed failure breakdown
    num_theta_fail = jnp.sum(theta_fail_mask)
    num_phi_fail = jnp.sum(phi_fail_mask)

    # Combined failure categories
    num_accuracy_fail = jnp.sum(~acc_mask)  # Failed due to position OR rotation
    num_limit_fail = jnp.sum(
        ~(theta_mask & phi_mask)
    )  # Failed due to theta OR phi limits

    # Only accuracy failed (limits OK)
    num_accuracy_only_fail = jnp.sum(~acc_mask & theta_mask & phi_mask)

    # Only limits failed (accuracy OK)
    num_limit_only_fail = jnp.sum(acc_mask & ~(theta_mask & phi_mask))

    # Both accuracy and limits failed
    num_both_fail = jnp.sum(~acc_mask & ~(theta_mask & phi_mask))

    failure_stats = {
        "total_samples": int(total_samples),
        "num_success": int(num_success),
        "num_fail": int(num_fail),
        "success_rate": float(jnp.mean(mask) * 100.0),
        # Individual failure types
        "num_theta_fail": int(num_theta_fail),
        "num_phi_fail": int(num_phi_fail),
        # Combined failure categories
        "num_accuracy_fail": int(num_accuracy_fail),
        "num_limit_fail": int(num_limit_fail),
        "num_accuracy_only_fail": int(num_accuracy_only_fail),
        "num_limit_only_fail": int(num_limit_only_fail),
        "num_both_fail": int(num_both_fail),
        # Percentages
        "theta_fail_rate": float(num_theta_fail / total_samples * 100),
        "phi_fail_rate": float(num_phi_fail / total_samples * 100),
        "accuracy_fail_rate": float(num_accuracy_fail / total_samples * 100),
        "limit_fail_rate": float(num_limit_fail / total_samples * 100),
        "accuracy_only_fail_rate": float(num_accuracy_only_fail / total_samples * 100),
        "limit_only_fail_rate": float(num_limit_only_fail / total_samples * 100),
        "both_fail_rate": float(num_both_fail / total_samples * 100),
    }

    return (
        jnp.mean(mask) * 100.0,
        jnp.mean(error[mask]),
        failure_stats,
    )


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


def eval_ik_with_no_coll(
    robot: CCRobot,
    eval_num: int,
    batched_ik_solve: Callable[[Array, Array], Array],
    batched_fk: Callable[[ConstantCurvatureState], Array],
):
    """Main function for basic IK."""
    num_sections = robot.config.num_sections

    print(f"start solve ik of num sections {num_sections}, num eval {eval_num}")

    # sample target transforms
    initial_states = sample_states_test(robot, eval_num)
    print(f"finish sample {eval_num} states, start forward")
    target_transforms = batched_fk(initial_states)
    tip_transform = jaxlie.SE3.from_matrix(target_transforms[:, -1, ...])
    target_wxyz = tip_transform.rotation().wxyz
    target_position = tip_transform.translation()

    # warmup
    print(f"finish forward, start warmup")
    jax.block_until_ready(batched_ik_solve(target_wxyz, target_position))

    # solve ik
    start = time.time()
    print("start solve ik")
    solution = batched_ik_solve(target_wxyz, target_position)
    jax.block_until_ready(solution)
    total_time = time.time() - start

    # get solved tip transforms
    fk_result = robot.forward_kinematics(solution)
    tip_transforms = jaxlie.SE3.from_matrix(fk_result[:, -1, ...])

    # compute metric
    metric = ik_metric(robot, solution, tip_transforms, target_position, target_wxyz)
    print(f"finish solve ik of num sections {num_sections}, total time: {total_time}s")
    print(f"success rate: {metric[0]:.2f}%")
    print(f"error: {metric[1]}")

    # Print detailed failure analysis
    failure_stats = metric[2]
    print("\n--- Detailed Failure Analysis ---")
    print(f"Total samples: {failure_stats['total_samples']}")
    print(
        f"Successful: {failure_stats['num_success']} ({failure_stats['success_rate']:.2f}%)"
    )
    print(
        f"Failed: {failure_stats['num_fail']} ({100 - failure_stats['success_rate']:.2f}%)"
    )

    error = metric[1]
    success_rate = round(metric[0], 2)

    return {
        "eval num": eval_num,
        "num sections": num_sections,
        "error": error,
        "success rate": success_rate,
        "total time": total_time,
        "failure_stats": failure_stats,
    }


def eval_ik_all_sections(
    robot_config_path: str, section_list: list, eval_num_list: list, eval_type: str
):
    all_results_summary = []
    for num_sections in section_list:
        config = json.load(open(robot_config_path))
        config["num_sections"] = num_sections
        if eval_type == "cc":
            robot = CCRobot.from_config(config)
        elif eval_type == "cc_extend":
            robot = CCRobotExtend.from_config(config)
        else:
            raise ValueError(f"Invalid eval type: {eval_type}")
        batched_fk = jax.vmap(robot._forward_kinematics)
        solver = IKSolver(
            robot, num_seeds_init=128, num_seeds_final=8, total_steps=200, init_steps=10
        )
        batched_ik_solve = jax.vmap(jax.jit(solver.solve_ik_best))
        for eval_num in eval_num_list:
            all_results_summary.append(
                eval_ik_with_no_coll(robot, eval_num, batched_ik_solve, batched_fk)
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
    section_list = [3]
    eval_num_list = [1000]
    robot_config_path = "configs/robots/cc_eval.json"
    eval_type = "cc"
    eval_ik_all_sections(robot_config_path, section_list, eval_num_list, eval_type)
