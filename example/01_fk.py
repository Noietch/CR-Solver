import jax
import jax.numpy as jnp

from soul.robots.cc_robot import CCRobot, ConstantCurvatureState
from soul.visualization.visualizer_plot import (
    visualize_cc_model_2d,
    visualize_cc_model_3d,
)

DISABLE_JIT = False

if DISABLE_JIT:
    import os

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)

robot = CCRobot.from_config("configs/robots/cc.json")

batch_state = ConstantCurvatureState(
    base_position=jnp.array([[0, 0, 0], [0, 0, 0]]),
    theta=jnp.array([[1, 2, 3], [0, 0, 0]]),
    phi=jnp.array([[-1, -2, -3], [0, 0, 0]]),
    length=jnp.array([[1, 1, 1], [1, 1, 1]]),
)

state = ConstantCurvatureState(
    base_position=jnp.array([0, 0, 0]),
    theta=jnp.array([3.1415926 / 3, 3.1415926 / 3, 3.1415926 / 3]),
    phi=jnp.array([0, 0, 0]),
    length=jnp.array([1, 1, 1]),
)

pose = robot.forward_kinematics(state)

visualize_cc_model_2d(
    pose,
    num_points=robot.config.num_points_per_section,
    save_path="visualization/forward_kinematics_2d.png",
)
visualize_cc_model_3d(
    pose,
    num_points=robot.config.num_points_per_section,
    save_path="visualization/forward_kinematics_3d.png",
)
