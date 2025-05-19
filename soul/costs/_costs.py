import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxls import Cost, Var, VarValues

from ..robots.pcc_robot import PCCRobot


@Cost.create_factory
def pose_cost(
    vals: VarValues,
    robot: PCCRobot,
    joint_var: Var[Array],
    target_pose: jaxlie.SE3,
    pos_weight: Array | float,
    ori_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses to target poses."""
    joint_cfg = vals[joint_var]
    Ts_point_world = robot._forward_kinematics(joint_cfg)
    pose_actual = jaxlie.SE3.from_matrix(Ts_point_world[-1, :, :])
    residual = (pose_actual.inverse() @ target_pose).log()
    pos_residual = residual[..., :3] * pos_weight
    ori_residual = residual[..., 3:] * ori_weight
    return jnp.concatenate([pos_residual, ori_residual]).flatten()


@Cost.create_factory
def position_cost(
    vals: VarValues,
    robot: PCCRobot,
    joint_var: Var[Array],
    target_position: jax.Array,
    pos_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses to target poses."""
    joint_cfg = vals[joint_var]
    Ts_point_world = robot._forward_kinematics(joint_cfg)
    pos_residual = jnp.linalg.norm(Ts_point_world[-1, :, :3] - target_position, axis=-1) * pos_weight
    return pos_residual.flatten()


@Cost.create_factory
def limit_cost(
    vals: VarValues,
    robot: PCCRobot,
    joint_var: Var[Array],
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing joint limit violations."""
    joint_cfg = vals[joint_var]
    residual_upper_kappa = jnp.maximum(0.0, joint_cfg.kappa - robot.config.upper_limits_kappa)
    residual_lower_kappa = jnp.maximum(0.0, robot.config.lower_limits_kappa - joint_cfg.kappa)
    residual_upper_phi = jnp.maximum(0.0, joint_cfg.phi - robot.config.upper_limits_phi)
    residual_lower_phi = jnp.maximum(0.0, robot.config.lower_limits_phi - joint_cfg.phi)
    return ((residual_upper_kappa + residual_lower_kappa + residual_upper_phi + residual_lower_phi) * weight).flatten()