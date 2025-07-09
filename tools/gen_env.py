import jax.numpy as jnp
import viser
from soul.robots.cc_robot import CCRobot
from soul.visualization.visualizer_viser import ViserSoftRobot, ViserWorld
from soul.geom import WorldCollision, RobotCollision
import time


def main():
    robot = CCRobot.from_config("configs/robots/cc_mobile_z.json")
    robot_coll = RobotCollision.from_config("configs/robots/cc_mobile_z.json")
    world_coll = WorldCollision.from_config("configs/maps/obstacles_03.json")
    # jax.debug.breakpoint()
    config_path = "configs/maps/obstacles_N.json"
     # Setup Visualization
    server = viser.ViserServer()
    robot_vis = ViserSoftRobot(server, robot_coll, root_node_name="/robot")
    robot_vis.create_sphere_visualizations()
    obstacles_vis = ViserWorld(
        server, world_coll, is_handle_able=True, config_path=config_path
    )
    obstacles_vis.create_mesh_visualizations()

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    main()
