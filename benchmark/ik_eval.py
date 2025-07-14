import jax
from jaxtyping import Array
import jax.numpy as jnp
import jaxlie
import time
import json
import os
from typing import Callable
from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.solver import IKSolver

DISABLE_JIT = False

if DISABLE_JIT:
    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def ik_metric(
    result_transform: jaxlie.SE3,
    target_position: Array,
    target_orientation: Array,
) -> float:
    result_position = result_transform.translation()
    result_orientation = result_transform.rotation()

    position_error = jnp.linalg.norm(result_position - target_position, axis=-1)
    position_threshold: float = 0.01
    rotation_threshold: float = 0.01

    orientation_error = jnp.linalg.norm(
        jnp.array(
            (jaxlie.SO3(target_orientation).inverse() @ result_orientation).log()
        ),
        axis=-1,
    )

    success_mask = jnp.logical_and(
        position_error < position_threshold,
        orientation_error < rotation_threshold,
    )

    return (
        jnp.mean(success_mask) * 100.0,
        jnp.mean(position_error[success_mask]),
        jnp.mean(orientation_error[success_mask]),
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
    metric = ik_metric(tip_transforms, target_position, target_wxyz)
    print(f"finish solve ik of num sections {num_sections}, total time: {total_time}s")
    print(f"success rate: {metric[0]:.2f}%")
    print(f"position error: {metric[1]}m")
    print(f"rotation error: {metric[2]}rad")
    position_error = metric[1]
    rotation_error = metric[2]
    success_rate = round(metric[0], 2)

    return {
        "eval num": eval_num,
        "num sections": num_sections,
        "position error": position_error,
        "rotation error": rotation_error,
        "success rate": success_rate,
        "total time": total_time,
    }


def eval_ik_all_sections(section_list: list, eval_num_list: list):
    all_results_summary = []
    for num_sections in section_list:
        config = json.load(open(f"configs/robots/cc_eval.json"))
        config["num_sections"] = num_sections
        robot = CCRobot.from_config(config)
        batched_fk = jax.vmap(robot._forward_kinematics)
        solver = IKSolver(
            robot, num_seeds_init=64, num_seeds_final=4, total_steps=1000, init_steps=10
        )
        batched_ik_solve = jax.vmap(jax.jit(solver.solve_ik_best))
        for eval_num in eval_num_list:
            all_results_summary.append(
                eval_ik_with_no_coll(robot, eval_num, batched_ik_solve, batched_fk)
            )

    print("\n\n--- IK test resume ---")
    header = f"{'num sections':<10} | {'eval num':<15} | {'position error':<15} | {'rotation error':<15} | {'success rate (%)':<15} | {'total time (s)':<18} "
    print(header)
    print("-" * len(header))
    for res_item in all_results_summary:
        eval_num_str = f"{res_item['eval num']}"
        ps_error_str = f"{res_item['position error']}"
        rt_error_str = f"{res_item['rotation error']}"
        sr_str = f"{res_item['success rate']:.2f}"
        time_str = f"{res_item['total time']:.3f}"
        print(
            f"{res_item['num sections']:<10} | {eval_num_str:<15} | {ps_error_str:<15} | {rt_error_str:<15} | {sr_str:<15} | {time_str:<18}"
        )


if __name__ == "__main__":
    test_list = [2, 3, 4, 5, 6]
    eval_num_list = [1000]
    eval_ik_all_sections(test_list, eval_num_list)
