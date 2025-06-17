import jax
import numpy as np
import jax.numpy as jnp

from soul.robots.pcc_robot import PCCRobot, ConstantCurvatureState
from soul.collision import RobotCollision
from soul.solver.graph import GraphPlanner
from soul.collision import Sphere
from soul.visualization.visualizer_plot import visualize_pcc_model_3d

DISABLE_JIT = False

if DISABLE_JIT:
    import os
    import jax

    os.environ["JAX_DISABLE_JIT"] = "True"
    jax.config.update("jax_disable_jit", True)


def test_random_sample_nodes(robot: PCCRobot, graph_planner: GraphPlanner, num_nodes: int):
    states = graph_planner.get_random_sampled_states(num_nodes)
    poses = robot.forward_kinematics(states)
    visualize_pcc_model_3d(
        poses,
        num_points=robot.config.num_points_per_section,
        save_path="visualization/random_sampled_nodes.png",
    )
    return states

def test_biased_sample_nodes(robot: PCCRobot, graph_planner: GraphPlanner, num_nodes: int):
    start_cfg = ConstantCurvatureState(
        base_position=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        kappa=jnp.array([-0.13385129, 1.0690628, 0.45618892], dtype=jnp.float32),
        phi=jnp.array([1.3893168, 0.34500587, -1.9371563], dtype=jnp.float32),
    )
    end_cfg = ConstantCurvatureState(
        base_position=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        kappa=jnp.array([-0.19718027, 1.3294444, 0.87349164], dtype=jnp.float32),
        phi=jnp.array([-1.8368645, -0.25360936, 2.550133], dtype=jnp.float32),
    )
    states = graph_planner.get_biased_sampled_states(start_cfg, end_cfg, num_nodes)
    poses = robot.forward_kinematics(states)
    visualize_pcc_model_3d(
        poses,
        num_points=robot.config.num_points_per_section,
        save_path="visualization/biased_sampled_nodes.png",
    )
    return states


def main():
    robot = PCCRobot.from_config("configs/robots/pcc.json")
    robot_coll = RobotCollision.from_config(
        "configs/robots/pcc.json", self_collision_sampling_rate=1
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.2])
    )
    graph_planner = GraphPlanner(robot, robot_coll, [sphere_coll])
    
    # test_random_sample_nodes(robot, graph_planner, 3)
    # test_biased_sample_nodes(robot, graph_planner, 10)

if __name__ == "__main__":
    main()