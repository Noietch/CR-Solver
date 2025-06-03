import jax
import jax.numpy as jnp
import jaxlie
import time

from soul.robots.pcc_robot import PCCRobot
from soul.solver import IKSolver

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def ik_metric(result_transform: jaxlie.SE3, target_position: jax.Array, target_orientation: jax.Array) -> float:
    result_position = result_transform.translation()
    result_orientation = result_transform.rotation()

    position_error = jnp.linalg.norm(result_position - target_position, axis=-1)
    position_threshold: float = 0.001
    rotation_threshold: float = 0.05
    
    orientation_error = jnp.linalg.norm(
        jnp.array(
            (
                jaxlie.SO3(target_orientation).inverse() @ result_orientation
            ).log()
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




def eval_ik_with_no_coll():
    """Main function for basic IK."""
    robot = PCCRobot.from_config("configs/robots/pcc_2d.json")
    
    solver = IKSolver(robot, num_seeds_init=64, num_seeds_final=4, total_steps=16, init_steps=6)
    batched_ik_solve = jax.vmap(jax.jit(solver.solve_ik))

    # sample target transforms
    eval_num = 10
    initial_states = solver.sample_states(eval_num)
    target_transforms = robot.forward_kinematics(initial_states)
    tip_transform = jaxlie.SE3.from_matrix(target_transforms[:, -1, ...])
    target_wxyz = tip_transform.rotation().wxyz
    target_position = tip_transform.translation()
    # warmup
    jax.block_until_ready(batched_ik_solve(target_wxyz, target_position))

    # solve ik
    start = time.time()
    solution = batched_ik_solve(target_wxyz, target_position)
    jax.block_until_ready(solution)
    total_time = (time.time() - start) / target_wxyz.shape[0]
    
    # get solved tip transforms
    fk_result = robot.forward_kinematics(solution)
    tip_transforms = jaxlie.SE3.from_matrix(fk_result[:, -1, ...])

    # compute metric
    metric = ik_metric(tip_transforms, target_position, target_wxyz)
    print(f"finish solve ik, total time: {total_time}s")
    print(f"success rate: {metric[0]:.2f}%")
    print(f"position error: {metric[1]}m")
    print(f"rotation error: {metric[2]}rad")


if __name__ == "__main__":
    eval_ik_with_no_coll()