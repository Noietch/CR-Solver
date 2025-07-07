import jax
from jaxtyping import Float, Array
import jax.numpy as jnp
import jaxlie
import time
import json
import os
from typing import Callable, List, Sequence
from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.solver import IKSolver
from soul.geom.collision_cc_robot import RobotCollision
from soul.geom.geometry import CollGeom, Sphere, cat_geoms
from soul.geom.collision import colldist_from_sdf

from benchmark.visualizer_eval import create_figure, visualizer_forward_samples

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


# [ (center_x, center_y, center_z, radius), ... ]
COLLISION_OBSTACLES = [
    (jnp.array([-1.0, -1.0, 1.5]), 0.15),
    (jnp.array([-1.0, -1.0, 2.5]), 0.15),
    (jnp.array([-1.0, -1.0, 3.5]), 0.15),
    (jnp.array([-1.0, 0.0, 1.5]), 0.15),
    (jnp.array([-1.0, 0.0, 2.5]), 0.15),
    (jnp.array([-1.0, 0.0, 3.5]), 0.15),
    (jnp.array([-1.0, 1.0, 1.5]), 0.15),
    (jnp.array([-1.0, 1.0, 2.5]), 0.15),
    (jnp.array([-1.0, 1.0, 3.5]), 0.15),
    (jnp.array([0.0, -1.0, 1.5]), 0.15),
    (jnp.array([0.0, -1.0, 2.5]), 0.15),
    (jnp.array([0.0, -1.0, 3.5]), 0.15),
    (jnp.array([0.0, 0.0, 1.5]), 0.15),
    (jnp.array([0.0, 0.0, 2.5]), 0.15),
    (jnp.array([0.0, 0.0, 3.5]), 0.15),
    (jnp.array([0.0, 1.0, 1.5]), 0.15),
    (jnp.array([0.0, 1.0, 2.5]), 0.15),
    (jnp.array([0.0, 1.0, 3.5]), 0.15),
    (jnp.array([1.0, -1.0, 1.5]), 0.15),
    (jnp.array([1.0, -1.0, 2.5]), 0.15),
    (jnp.array([1.0, -1.0, 3.5]), 0.15),
    (jnp.array([1.0, 0.0, 1.5]), 0.15),
    (jnp.array([1.0, 0.0, 2.5]), 0.15),
    (jnp.array([1.0, 0.0, 3.5]), 0.15),
    (jnp.array([1.0, 1.0, 1.5]), 0.15),
    (jnp.array([1.0, 1.0, 2.5]), 0.15),
    (jnp.array([1.0, 1.0, 3.5]), 0.15),
]


def ik_metric(
    result_transform: jaxlie.SE3,
    target_position: Array,
    target_orientation: Array,
) -> float:
    result_position = result_transform.translation()
    result_orientation = result_transform.rotation()

    position_error = jnp.linalg.norm(result_position - target_position, axis=-1)
    position_threshold: float = 0.001
    rotation_threshold: float = 0.05

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

    self_cost = 0.0

    # 3. Sum costs and check for collision
    total_cost = world_cost + self_cost
    return total_cost > 1e-6


def eval_ik_with_coll(
    robot: CCRobot,
    eval_num: int,
    batched_ik_solve: Callable[[Array, Array, Sequence[CollGeom]], Array],
    batched_fk: Callable[[ConstantCurvatureState], Array],
    robot_coll: RobotCollision,
    world_geom: CollGeom,
    visualize: bool = False,
    save_path: str = None,
):
    """Main function for basic IK with collision avoidance."""
    ax = create_figure()
    num_sections = robot.config.num_sections
    print(
        f"start solve ik WITH COLLISION of num sections {num_sections}, num eval {eval_num}"
    )

    # 采样无碰撞的目标点
    print(f"Sampling {eval_num} collision-free target states...")
    collision_free_states = []
    total_sampled = 0
    max_sampling_attempts = eval_num * 100  # Safety break for the while loop
    num_free_states = 0

    while num_free_states < eval_num and total_sampled < max_sampling_attempts:
        # 批量采样一批状态
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
    visualizer_forward_samples(
        ax,
        target_transforms,
        target_position,
        num_points=robot.config.num_points_per_section,
        save_path=save_path,
    )

    # warmup
    print(f"finish forward, start warmup")
    jax.block_until_ready(batched_ik_solve(target_wxyz, target_position, [world_geom]))

    # solve ik
    start = time.time()
    print("start solve ik")
    solution_states = batched_ik_solve(target_wxyz, target_position, [world_geom])
    jax.block_until_ready(solution_states)
    total_time = time.time() - start

    # Check if IK solutions have collision
    is_solution_collision_vmap = jax.vmap(
        is_state_in_collision, in_axes=(0, None, None, None)
    )

    solution_collision_mask = is_solution_collision_vmap(
        solution_states, robot, robot_coll, world_geom
    )  # True if collision

    # calculate accuracy metrics
    fk_result = robot.forward_kinematics(solution_states)
    tip_transforms = jaxlie.SE3.from_matrix(fk_result[:, -1, ...])
    # Combine accuracy and collision results to get final success criteria
    position_error = jnp.linalg.norm(
        tip_transforms.translation() - target_position,
        axis=-1,
    )
    orientation_error = jnp.linalg.norm(
        (jaxlie.SO3(target_wxyz).inverse() @ tip_transforms.rotation()).log(),
        axis=-1,
    )

    # Original success mask (accuracy only)
    accuracy_success_mask = jnp.logical_and(
        position_error < 0.5,
        orientation_error < 0.5,
    )

    # Final success mask (accurate and collision-free)
    final_success_mask = jnp.logical_and(
        accuracy_success_mask,
        ~solution_collision_mask,
    )

    final_success_rate = jnp.mean(final_success_mask) * 100.0
    final_pos_error = jnp.mean(position_error[final_success_mask])
    final_rot_error = jnp.mean(orientation_error[final_success_mask])

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


def eval_ik_all_sections(section_list: list, eval_num_list: list):
    all_results_summary = []
    for num_sections in section_list:
        config_path = f"configs/robots/cc_eval.json"
        config = json.load(open(config_path))
        config["num_sections"] = num_sections
        robot = CCRobot.from_config(config)
        # Pass the updated config dictionary, not the file path, to ensure consistency
        robot_coll = RobotCollision.from_config(config)

        # Create world geometry from the list of obstacles
        spheres: List[CollGeom] = [
            Sphere.from_center_and_radius(center=obs[0], radius=jnp.array(obs[1]))
            for obs in COLLISION_OBSTACLES
        ]
        world_geom = cat_geoms(spheres)

        batched_fk = jax.vmap(robot._forward_kinematics)
        solver = IKSolver(
            robot,
            num_seeds_init=64,
            num_seeds_final=4,
            total_steps=1000,
            init_steps=10,
            coll=robot_coll,
        )
        batched_ik_solve = jax.vmap(
            solver.solve_ik_best_with_coll, in_axes=(0, 0, None)
        )
        for eval_num in eval_num_list:
            all_results_summary.append(
                eval_ik_with_coll(
                    robot,
                    eval_num,
                    batched_ik_solve,
                    batched_fk,
                    robot_coll,
                    world_geom,
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


if __name__ == "__main__":
    test_list = [2, 3, 4, 5, 6]
    eval_num_list = [250]
    eval_ik_all_sections(test_list, eval_num_list)
