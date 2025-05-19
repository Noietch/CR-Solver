"""
Solves the basic IK problem.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
from soul.robots.pcc_robot import PCCRobot
from soul.costs import pose_cost, limit_cost, position_cost

def solve_ik(
    robot: PCCRobot,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: np.ndarray. Target orientation.
        target_position: np.ndarray. Target position.

    Returns:
        cfg: np.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_wxyz),
        jnp.array(target_position),
    )
    return cfg


@jdc.jit
def _solve_ik_jax(
    robot: PCCRobot,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    joint_var = robot.var_cls(0)
    factors = [
        position_cost(
            robot,
            joint_var,
            target_position,
            pos_weight=50.0,
        ),
        # pose_cost(
        #     robot,
        #     joint_var,
        #     jaxlie.SE3.from_rotation_and_translation(
        #         jaxlie.SO3(target_wxyz), target_position
        #     ),
        #     pos_weight=50.0,
        #     ori_weight=10.0,
        # ),
        limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]
