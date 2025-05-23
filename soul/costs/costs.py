import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxls import Cost, Var, VarValues

from ..robots.pcc_robot import PCCRobot, ConstantCurvatureState
from ..collision.pcc_robot_collision import RobotCollision
from ..collision.geometry import CollGeom
from ..collision.collision import colldist_from_sdf, collide


@Cost.create_factory
def pose_cost(
    vals: VarValues,
    robot: PCCRobot,
    robot_var: Var[ConstantCurvatureState],
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
    residual_upper_kappa = jnp.maximum(
        0.0, state.kappa - robot.config.upper_limits_kappa
    )
    residual_lower_kappa = jnp.maximum(
        0.0, robot.config.lower_limits_kappa - state.kappa
    )
    residual_upper_phi = jnp.maximum(0.0, state.phi - robot.config.upper_limits_phi)
    residual_lower_phi = jnp.maximum(0.0, robot.config.lower_limits_phi - state.phi)
    return (
        (
            residual_upper_kappa
            + residual_lower_kappa
            + residual_upper_phi
            + residual_lower_phi
        )
        * weight
    ).flatten()


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


@Cost.create_factory
def continuous_collision_cost(
    vals: VarValues,
    robot: PCCRobot,
    robot_coll: RobotCollision,
    world_coll_obj: CollGeom,
    prev_traj_vars: Var[ConstantCurvatureState],
    curr_traj_vars: Var[ConstantCurvatureState],
):
    coll = robot_coll.get_swept_capsules(
        robot, vals[prev_traj_vars], vals[curr_traj_vars]
    )
    dist = collide(coll.reshape(-1, 1), world_coll_obj.reshape(1, -1))
    colldist = colldist_from_sdf(dist, 0.1)
    return (colldist * 20.0).flatten()


@Cost.create_factory
def smoothness_cost(
    vals: VarValues,
    curr_robot_var: Var[ConstantCurvatureState],
    past_robot_var: Var[ConstantCurvatureState],
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing joint configuration differences (velocity)."""
    return ((vals[curr_robot_var] - vals[past_robot_var])).flatten() * weight


@Cost.create_factory
def start_end_similarity_cost(
    vals: VarValues,
    end_cfg: ConstantCurvatureState,
    start_cfg: ConstantCurvatureState,
    weight: Array | float,
) -> Array:
    """Computes the similarity between start and end configuration."""
    end_state = vals[end_cfg]
    start_state = vals[start_cfg]
    return ((end_state - start_state)).flatten() * weight


@Cost.create_factory
def boundary_cost(
    vals: VarValues,
    robot_var: Var[ConstantCurvatureState],
    start_end_cfg: ConstantCurvatureState,
    weight: Array | float,
) -> Array:
    jax.debug.breakpoint()
    """Computes the residual penalizing start and end pose differences."""
    state = vals[robot_var]
    return ((state - start_end_cfg)).flatten() * weight


@Cost.create_factory
def elastic_energy_cost(
    vals: VarValues,
    robot_var: Var[ConstantCurvatureState],
    weight: Array | float,
) -> Array:
    """Computes the elastic energy of the robot. Penalize large kappa."""
    kappa = vals[robot_var].kappa
    return (kappa * weight).flatten()


@Cost.create_factory
def rest_base_cost(
    vals: VarValues,
    robot_var: Var[ConstantCurvatureState],
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing the difference between the current state and the rest pose."""
    # TODO: This is a hack to get the rest pose of the robot. Need to fix this.
    state = vals[robot_var]
    return ((state.base_position)).flatten() * weight
