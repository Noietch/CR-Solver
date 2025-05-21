import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxls import Cost, Var, VarValues

from ..robots.pcc_robot import PCCRobot, ConstantCurvatureState
from ..collision.pcc_robot_collision import RobotCollision
from ..collision.geometry import CollGeom
from ..collision.collision import colldist_from_sdf

@Cost.create_factory
def pose_cost(
    vals: VarValues,
    robot: PCCRobot,
    robot_var: Var[Array],
    target_pose: jaxlie.SE3,
    pos_weight: Array | float,
    ori_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses to target poses."""
    state = vals[robot_var]
    robot_pose = robot._forward_kinematics(state)
    tip_pose = jaxlie.SE3.from_matrix(robot_pose[-1, ...])
    residual = (tip_pose.inverse() @ target_pose).log()
    pos_residual = residual[..., :3] * pos_weight
    ori_residual = residual[..., 3:] * ori_weight
    return jnp.concatenate([pos_residual, ori_residual]).flatten()


@Cost.create_factory
def limit_cost(
    vals: VarValues,
    robot: PCCRobot,
    robot_var: Var[ConstantCurvatureState],
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing joint limit violations."""
    state = vals[robot_var]
    residual_upper_kappa = jnp.maximum(0.0, state.kappa - robot.config.upper_limits_kappa)
    residual_lower_kappa = jnp.maximum(0.0, robot.config.lower_limits_kappa - state.kappa)
    residual_upper_phi = jnp.maximum(0.0, state.phi - robot.config.upper_limits_phi)
    residual_lower_phi = jnp.maximum(0.0, robot.config.lower_limits_phi - state.phi)
    return ((residual_upper_kappa + residual_lower_kappa + residual_upper_phi + residual_lower_phi) * weight).flatten()


@Cost.create_factory
def world_collision_cost(
    vals: VarValues,
    robot: PCCRobot,
    robot_coll: RobotCollision,
    robot_var: Var[ConstantCurvatureState],
    world_geom: CollGeom,
    margin: float,
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing world collisions below a margin."""
    state = vals[robot_var]
    dist_matrix = robot_coll.compute_world_collision_distance(robot, state, world_geom)
    residual = colldist_from_sdf(dist_matrix, margin)
    return (residual * weight).flatten()