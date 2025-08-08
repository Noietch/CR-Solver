import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxls import Cost, Var, VarValues

from ..robots.tdcr_robot import TDCRRobot
from ..robots.cc_robot import ConstantCurvatureState


# additional costs for tdcr
@Cost.create_factory
def tendon_length_velocity_limit_cost(
    vals: VarValues,
    robot: TDCRRobot,
    curr_robot_var: Var[ConstantCurvatureState],
    past_robot_var: Var[ConstantCurvatureState],
    dt: float,
    max_velocity: Array | float,
    weight: Array | float,
) -> Array:
    """
    Computes the residual penalizing tendon velocities that exceed maximum limits.
    This ensures tendon velocities stay within safe operating ranges.

    Args:
        vals: Variable values
        robot: TDCR robot instance
        curr_robot_var: Current robot state variable
        past_robot_var: Previous robot state variable
        dt: Time step between states
        max_velocity: Maximum allowed tendon velocity
        weight: Weight for the cost

    Returns:
        Weighted residual for tendon velocity limit violations
    """
    curr_state = vals[curr_robot_var]
    past_state = vals[past_robot_var]

    # Calculate tendon lengths for both states
    curr_tendon_lengths = robot.calculate_tendon_lengths(curr_state)
    past_tendon_lengths = robot.calculate_tendon_lengths(past_state)

    # Calculate velocities
    tendon_velocities = (curr_tendon_lengths - past_tendon_lengths) / dt

    # Penalize velocities exceeding the maximum (both positive and negative)
    residual = jnp.maximum(0.0, jnp.abs(tendon_velocities) - max_velocity)

    return (residual * weight).flatten()


@Cost.create_factory
def tendon_length_velocity_cost(
    vals: VarValues,
    robot: TDCRRobot,
    curr_robot_var: Var[ConstantCurvatureState],
    past_robot_var: Var[ConstantCurvatureState],
    dt: float,
    weight: Array | float,
) -> Array:
    """
    Computes the residual penalizing tendon length velocity.
    This ensures smooth tendon movements between time steps.

    Args:
        vals: Variable values
        robot: TDCR robot instance
        curr_robot_var: Current robot state variable
        past_robot_var: Previous robot state variable
        dt: Time step between states
        weight: Weight for the cost

    Returns:
        Weighted residual for tendon length velocity
    """
    curr_state = vals[curr_robot_var]
    past_state = vals[past_robot_var]

    # Calculate tendon lengths for both states
    curr_tendon_lengths = robot.calculate_tendon_lengths(curr_state)
    past_tendon_lengths = robot.calculate_tendon_lengths(past_state)

    # Calculate velocity (change in length / dt)
    tendon_velocities = (curr_tendon_lengths - past_tendon_lengths) / dt

    # Return weighted velocity residual
    return (tendon_velocities * weight).flatten()


@Cost.create_factory
def tendon_length_acceleration_cost(
    vals: VarValues,
    robot: TDCRRobot,
    prev_robot_var: Var[ConstantCurvatureState],
    curr_robot_var: Var[ConstantCurvatureState],
    next_robot_var: Var[ConstantCurvatureState],
    dt: float,
    weight: Array | float,
) -> Array:
    """
    Computes the residual penalizing tendon length acceleration.
    This ensures smooth acceleration profiles for tendon movements.

    Args:
        vals: Variable values
        robot: TDCR robot instance
        prev_robot_var: Previous robot state variable (t-1)
        curr_robot_var: Current robot state variable (t)
        next_robot_var: Next robot state variable (t+1)
        dt: Time step between states
        weight: Weight for the cost

    Returns:
        Weighted residual for tendon length acceleration
    """
    prev_state = vals[prev_robot_var]
    curr_state = vals[curr_robot_var]
    next_state = vals[next_robot_var]

    # Calculate tendon lengths for all three states
    prev_tendon_lengths = robot.calculate_tendon_lengths(prev_state)
    curr_tendon_lengths = robot.calculate_tendon_lengths(curr_state)
    next_tendon_lengths = robot.calculate_tendon_lengths(next_state)

    # Calculate velocities
    vel_curr = (curr_tendon_lengths - prev_tendon_lengths) / dt
    vel_next = (next_tendon_lengths - curr_tendon_lengths) / dt

    # Calculate acceleration (change in velocity / dt)
    tendon_accelerations = (vel_next - vel_curr) / dt

    # Return weighted acceleration residual
    return (tendon_accelerations * weight).flatten()


@Cost.create_factory
def tendon_length_jerk_cost(
    vals: VarValues,
    robot: TDCRRobot,
    robot_vars: list[Var[ConstantCurvatureState]],
    dt: float,
    weight: Array | float,
) -> Array:
    """
    Computes the residual penalizing tendon length jerk (rate of change of acceleration).
    This ensures very smooth tendon movements with minimal jerky motion.

    Args:
        vals: Variable values
        robot: TDCR robot instance
        robot_vars: List of 4 consecutive robot state variables [t-1, t, t+1, t+2]
        dt: Time step between states
        weight: Weight for the cost

    Returns:
        Weighted residual for tendon length jerk
    """
    assert len(robot_vars) == 4, "Jerk cost requires 4 consecutive states"

    # Get states
    states = [vals[var] for var in robot_vars]

    # Calculate tendon lengths for all states
    tendon_lengths = [robot.calculate_tendon_lengths(state) for state in states]

    # Calculate accelerations at t and t+1
    acc_t = (
        (tendon_lengths[2] - tendon_lengths[1])
        - (tendon_lengths[1] - tendon_lengths[0])
    ) / (dt * dt)
    acc_t1 = (
        (tendon_lengths[3] - tendon_lengths[2])
        - (tendon_lengths[2] - tendon_lengths[1])
    ) / (dt * dt)

    # Calculate jerk (change in acceleration / dt)
    tendon_jerk = (acc_t1 - acc_t) / dt

    # Return weighted jerk residual
    return (tendon_jerk * weight).flatten()
