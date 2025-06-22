from soul.envs.obs_env import ObstacleEnv


if __name__ == "__main__":
    env = ObstacleEnv("configs/maps/obstacles.json")
    env.show(editor=True)