import jax
import jax.numpy as jnp

from soul.robots.pcc_robot import PCCRobot, ConstantCurvatureState
from soul.visualization.visualizer_plot import visualize_pcc_model_2d

jax.config.update("jax_disable_jit", True)

robot = PCCRobot.from_config("configs/pcc_2d.json")

batch_state = ConstantCurvatureState(
    base_position=jnp.array([[0, 0, 0], [0, 0, 0]]),
    kappa=jnp.array([[1, 2, 3], [0, 0, 0]]),
    phi=jnp.array([[-1, -2, -3], [0, 0, 0]]),
)

state = ConstantCurvatureState(
    base_position=jnp.array([0, 0, 0]),
    kappa=jnp.array([1, 2, 3]),
    phi=jnp.array([0, 0, 0]),
)

pose = robot.forward_kinematics(state)
visualize_pcc_model_2d(pose, num_points=robot.config.num_points_per_section, save_path="visualization/forward_kinematics.png")