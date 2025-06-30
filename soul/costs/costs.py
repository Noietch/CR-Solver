import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxls import Cost, Var, VarValues

from ..robots.pcc_robot import PCCRobot, ConstantCurvatureState
from ..geom.collision_pcc_robot import RobotCollision
from ..geom.geometry import CollGeom
from ..geom.collision import colldist_from_sdf, collide


@Cost.create_factory
def position_cost(
    vals: VarValues,
    robot: PCCRobot,
    robot_var: Var[ConstantCurvatureState],
    target_position: Array,
    weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses to target poses."""
    state = vals[robot_var]
    robot_pose = robot._forward_kinematics(state)
    tip_position = robot_pose[-1, :3, 3]
    residual = tip_position - target_position[:3, 3]
    return (residual * weight).flatten()


@Cost.create_factory
def shape_cost(
    vals: VarValues,
    robot: PCCRobot,
    robot_var: Var[ConstantCurvatureState],
    target_shape: Array,
    weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses to target poses."""
    state = vals[robot_var]
    robot_pose = robot._forward_kinematics(state)  # (N, 4, 4)
    assert robot_pose.shape == target_shape.shape
    residual = robot_pose[..., :3, 3] - target_shape[..., :3, 3]
    return (residual * weight).flatten()


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
    residual_upper_theta = jnp.maximum(
        0.0, state.theta - robot.config.upper_limits_theta
    )
    residual_lower_theta = jnp.maximum(
        0.0, robot.config.lower_limits_theta - state.theta
    )
    residual_upper_phi = jnp.maximum(0.0, state.phi - robot.config.upper_limits_phi)
    residual_lower_phi = jnp.maximum(0.0, robot.config.lower_limits_phi - state.phi)
    return (
        (
            residual_upper_theta
            + residual_lower_theta
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
def self_collision_cost(
    vals: VarValues,
    robot: PCCRobot,
    robot_coll: RobotCollision,
    joint_var: Var[Array],
    margin: float,
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing self-collisions below a margin."""
    cfg = vals[joint_var]
    active_distances = robot_coll.compute_self_collision_distance(robot, cfg)
    residual = colldist_from_sdf(active_distances, margin)
    return (residual * weight).flatten()


@Cost.create_factory
def continuous_collision_cost(
    vals: VarValues,
    robot: PCCRobot,
    robot_coll: RobotCollision,
    world_coll_obj: CollGeom,
    prev_traj_vars: Var[ConstantCurvatureState],
    curr_traj_vars: Var[ConstantCurvatureState],
    weight: Array | float,
):
    coll = robot_coll.get_swept_capsules(
        robot, vals[prev_traj_vars], vals[curr_traj_vars]
    )
    dist = collide(coll.reshape(-1, 1), world_coll_obj.reshape(1, -1))
    colldist = colldist_from_sdf(dist, 0.05)
    return (colldist * weight).flatten()


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
    """Computes the residual penalizing start and end pose differences."""
    state = vals[robot_var]
    return ((state - start_end_cfg)).flatten() * weight


@Cost.create_factory
def elastic_energy_cost(
    vals: VarValues,
    robot_var: Var[ConstantCurvatureState],
    weight: Array | float,
) -> Array:
    """Computes the elastic energy of the robot. Penalize large theta."""
    theta = vals[robot_var].theta
    return (theta * weight).flatten()


@Cost.create_factory
def rest_base_cost(
    vals: VarValues,
    robot_var: Var[ConstantCurvatureState],
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing the difference between the current state and the rest pose."""
    state = vals[robot_var]
    return ((state.base_position)).flatten() * weight


@Cost.create_factory
def trajectory_length_cost(
    vals: VarValues,
    robot: PCCRobot,
    curr_robot_var: Var[ConstantCurvatureState],
    past_robot_var: Var[ConstantCurvatureState],
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing the total path length of the end-effector."""
    prev_states = vals[past_robot_var]
    curr_states = vals[curr_robot_var]
    prev_poses = robot.forward_kinematics(prev_states)
    curr_poses = robot.forward_kinematics(curr_states)
    prev_ee_positions = prev_poses[-1, :3, 3]
    curr_ee_positions = curr_poses[-1, :3, 3]
    diffs = curr_ee_positions - prev_ee_positions
    dists = jnp.linalg.norm(diffs, axis=-1)
    return (dists * weight).flatten()


# --- Finite Difference Costs (Velocity, Acceleration, Jerk) ---


@Cost.create_factory
def five_point_velocity_cost(
    vals: VarValues,
    robot: PCCRobot,  # Needed for limits
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing velocity limit violations (5-point stencil)."""
    q_tm2 = vals[var_t_minus_2].theta
    q_tm1 = vals[var_t_minus_1].theta
    q_tp1 = vals[var_t_plus_1].theta
    q_tp2 = vals[var_t_plus_2].theta

    velocity = (-q_tp2 + 8 * q_tp1 - 8 * q_tm1 + q_tm2) / (12 * dt)
    vel_limits = 0
    limit_violation = jnp.maximum(0.0, jnp.abs(velocity) - vel_limits)
    return (limit_violation * weight).flatten()


@Cost.create_factory
def five_point_acceleration_cost(
    vals: VarValues,
    var_t: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    weight: Array | float,
) -> Array:
    """Computes the residual minimizing joint acceleration (5-point stencil)."""
    q_tm2 = vals[var_t_minus_2].theta
    q_tm1 = vals[var_t_minus_1].theta
    q_t = vals[var_t].theta
    q_tp1 = vals[var_t_plus_1].theta
    q_tp2 = vals[var_t_plus_2].theta

    acceleration = (-q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2) / (12 * dt**2)
    return (acceleration * weight).flatten()


@Cost.create_factory
def five_point_jerk_cost(
    vals: VarValues,
    var_t_plus_3: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    var_t_minus_3: Var[Array],
    dt: float,
    weight: Array | float,
) -> Array:
    """Computes the residual minimizing joint jerk (7-point stencil)."""
    q_tm3 = vals[var_t_minus_3].theta
    q_tm2 = vals[var_t_minus_2].theta
    q_tm1 = vals[var_t_minus_1].theta
    q_tp1 = vals[var_t_plus_1].theta
    q_tp2 = vals[var_t_plus_2].theta
    q_tp3 = vals[var_t_plus_3].theta

    jerk = (-q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3) / (
        8 * dt**3
    )
    return (jerk * weight).flatten()
