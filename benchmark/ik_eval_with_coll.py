import jax
from jaxtyping import Array
import numpy as np
import jax.numpy as jnp
import jaxlie
import time
import json
import os
from typing import Callable, Sequence
from soul.robots.cc_robot_extend import CCRobot, ConstantCurvatureState
from soul.solver import IKSolver
from soul.geom import (
    RobotCollision,
    WorldCollision,
    CollGeom,
    colldist_from_sdf,
)
from soul.visualization.visualizer_plot import visualize_cc_model_3d

jax.config.update("jax_default_matmul_precision", "highest")

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def ik_metric_with_coll(
    result_transform: jaxlie.SE3,
    target_position: Array,
    target_orientation: Array,
    solution_collision_mask: Array,
) -> tuple[float, float, float]:
    """
    Calculates IK success metrics considering both accuracy and collision avoidance.

    Args:
        result_transform: The resulting end-effector transforms from the IK solver.
        target_position: The target positions.
        target_orientation: The target orientations (as wxyz quaternions).
        solution_collision_mask: A boolean array where True indicates a collision.

    Returns:
        A tuple containing:
        - final_success_rate: The percentage of solutions that are both accurate and collision-free.
        - final_pos_error: The mean position error for successful solutions.
        - final_rot_error: The mean rotation error for successful solutions.
    """
    # Accuracy thresholds from the evaluation function
    position_threshold: float = 0.03
    rotation_threshold: float = 0.03

    position_error = jnp.linalg.norm(
        result_transform.translation() - target_position,
        axis=-1,
    )
    orientation_error = jnp.linalg.norm(
        (jaxlie.SO3(target_orientation).inverse() @ result_transform.rotation()).log(),
        axis=-1,
    )

    # Original success mask (accuracy only)
    accuracy_success_mask = jnp.logical_and(
        position_error < position_threshold,
        orientation_error < rotation_threshold,
    )

    # Final success mask (accurate and collision-free)
    final_success_mask = jnp.logical_and(
        accuracy_success_mask,
        ~solution_collision_mask,
    )

    # Calculate statistics for different failure modes
    num_total = len(accuracy_success_mask)
    num_accuracy_fail = num_total - jnp.sum(accuracy_success_mask)
    num_collision_fail = jnp.sum(solution_collision_mask)
    num_accuracy_only_success = jnp.sum(accuracy_success_mask) - jnp.sum(
        final_success_mask
    )

    print(f"\nFailure Analysis:")
    print(f"Total samples: {num_total}")
    print(
        f"Failed due to accuracy: {num_accuracy_fail} ({num_accuracy_fail/num_total*100:.1f}%)"
    )
    print(
        f"Failed due to collision: {num_collision_fail} ({num_collision_fail/num_total*100:.1f}%)"
    )
    print(
        f"Accurate but in collision: {num_accuracy_only_success} ({num_accuracy_only_success/num_total*100:.1f}%)"
    )

    final_success_rate = jnp.mean(final_success_mask) * 100.0
    # Use jnp.nanmean to avoid errors if no solutions are successful
    final_pos_error = jnp.nan_to_num(jnp.mean(position_error[final_success_mask]))
    final_rot_error = jnp.nan_to_num(jnp.mean(orientation_error[final_success_mask]))

    return final_success_rate, final_pos_error, final_rot_error


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

    length = jax.random.uniform(
        key=subkey,
        shape=(num_states, robot.config.num_sections),
        minval=robot.config.lower_limits_length,
        maxval=robot.config.upper_limits_length,
    )

    states = ConstantCurvatureState(
        base_position=jnp.zeros((num_states, 3)),
        theta=theta,
        phi=phi,
        length=length,
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
    max_sampling_attempts = eval_num * 100  # Safety break for the while loop
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
    target_transforms = batched_fk(initial_states)
    tip_transform = jaxlie.SE3.from_matrix(target_transforms[:, -1, ...])
    target_wxyz = tip_transform.rotation().wxyz
    target_position = tip_transform.translation()

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
        np.savez(
            save_path,
            target_position=np.array(target_position),
            target_wxyz=np.array(target_wxyz),
            fk_result=np.array(fk_result),
            initial_states_theta=np.array(initial_states.theta),
            initial_states_phi=np.array(initial_states.phi),
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
    final_success_rate, final_pos_error, final_rot_error = ik_metric_with_coll(
        tip_transforms, target_position, target_wxyz, solution_collision_mask
    )

    print(f"--- With Collision Results ---")
    print(
        f"Final Success Rate (accurate AND collision-free): {final_success_rate:.2f}%"
    )
    print(f"Final Position Error: {final_pos_error:.3f}m")
    print(f"Final Rotation Error: {final_rot_error:.3f}rad")
    print(f"finish solve ik of num sections {num_sections}, total time: {total_time}s")

    return {
        "eval num": eval_num,
        "num sections": num_sections,
        "position error": final_pos_error,
        "rotation error": final_rot_error,
        "success rate": final_success_rate,
        "total time": total_time,
    }


def eval_ik_all_sections(
    robot_config_path: str,
    world_config_path: str,
    section_list: list,
    eval_num_list: list,
    save_dir: str,
):
    all_results_summary = []
    for num_sections in section_list:
        # load robot config
        config = json.load(open(robot_config_path))
        config["num_sections"] = num_sections
        robot = CCRobot.from_config(config)
        robot_coll = RobotCollision.from_config(config)

        # load world config
        world_coll = WorldCollision.from_config(world_config_path)

        batched_fk = jax.vmap(robot._forward_kinematics)
        solver = IKSolver(
            robot,
            num_seeds_init=128,
            num_seeds_final=8,
            total_steps=200,
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


def visualize_ik_with_coll(save_path: str, world_config_path: str):
    data = np.load(save_path)
    target_position = data["target_position"]
    target_wxyz = data["target_wxyz"]
    fk_result = data["fk_result"]
    # Randomly select 3 solutions
    num_solutions = len(fk_result)
    selected_indices = np.random.choice(num_solutions, size=3, replace=False)
    # Get the selected solutions
    fk_result = fk_result[selected_indices]
    target_position = target_position[selected_indices]
    visualize_cc_model_3d(
        pose=fk_result,
        target_position=target_position,
        world_coll_config=world_config_path,
        save_path=save_path.replace(".npz", "_fk.png"),
    )


if __name__ == "__main__":
    test_list = [3, 4, 5, 6]
    eval_num_list = [100]
    robot_config_path = "configs/robots/cc_extend_eval.json"
    world_config_path = "configs/maps/ik_maps/obstacles_lattice.json"
    result_dir = "results/ik_with_coll_lattice"
    eval_ik_all_sections(
        robot_config_path, world_config_path, test_list, eval_num_list, result_dir
    )
    visualize_ik_with_coll(
        f"{result_dir}/ik_with_coll_sections_4_eval_100.npz", world_config_path
    )
