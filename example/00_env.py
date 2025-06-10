from soul.envs.obs_env import ObstacleEnv


if __name__ == "__main__":
    env = ObstacleEnv("configs/maps/obstacles.json")
    env.visualize(editor=True, export_path="configs/maps/obstacles.json")
