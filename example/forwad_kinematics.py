import jax
import jax.numpy as jnp

from soul.robots.pcc_robot import PCCRobot, ConstantCurvatureState
from soul.visualization.visualizer import visualize_pcc_model_2d

jax.config.update("jax_disable_jit", True)

pcc_model_config = dict(
    num_sections=3,
    num_points_per_section=50,
    length=1.0,
    lower_limits_kappa=[-4, -4, -4],
    upper_limits_kappa=[4, 4, 4],
    lower_limits_phi=[0, 0, 0],
    upper_limits_phi=[0, 0, 0],
)

pcc_robot = PCCRobot.from_config(pcc_model_config)

batch_state = ConstantCurvatureState(
    kappa=jnp.array([[1, 2, 3], [0, 0, 0]]),
    phi=jnp.array([[-1, -2, -3], [0, 0, 0]]),
)

state = ConstantCurvatureState(
    kappa=jnp.array([1, 2, 4]),
    phi=jnp.array([0, 0, 0]),
)

pose = pcc_robot.forward_kinematics(state)
visualize_pcc_model_2d(pose, num_points=pcc_model_config["num_points_per_section"], save_path="visualization/forward_kinematics.png")