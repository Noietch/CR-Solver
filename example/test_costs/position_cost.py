import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxls import Var, VarValues
import numpy as np
from soul.robots.pcc_robot_array import PCCRobot
from soul.solver import solve_ik


def position_cost(
    joint_var: Var[Array],
    robot: PCCRobot,
    target_position: jax.Array,
    pos_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses to target poses."""
    joint_cfg = joint_var
    Ts_point_world = robot._forward_kinematics(joint_cfg)
    pose_actual = jaxlie.SE3.from_matrix(Ts_point_world[-1, :, :])
    pos_residual = pose_actual.translation() - target_position
    return pos_residual.sum() * pos_weight

def main():
    robot = PCCRobot.from_config("configs/pcc_2d.json")
    target_wxyz = np.array([1, 0, 0, 0])
    target_position = np.array([1, 0, 0])
    var = jnp.zeros(robot.config.num_sections * 2 + 3)
    cost, grad = jax.value_and_grad(position_cost)(var, robot, target_position, 1.0)
    print(cost)
    print(grad)

if __name__ == "__main__":
    main()
