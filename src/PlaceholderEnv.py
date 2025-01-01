import gym
from numpy import shape


class PlaceholderEnv:
    def __init__(self, env_name='CartPole-v0', *args, **kwargs):
        self.env = gym.make(env_name)

if __name__ == "__main__":
    env = PlaceholderEnv().env