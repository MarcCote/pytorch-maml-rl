import gym
from functools import partial


def _make_env(env_name, seed):
    env = gym.make(env_name) #env = gym.make(env_name, **env_kwargs) -- we don't have kwargs for tw-env
    if hasattr(env, 'seed'):
        env.seed(seed)
    return env


def make_env(env_name, seed=None): #def make_env(env_name, env_kwargs={}, seed=None): - we don't have kwargs for tw-env
    return partial(_make_env, env_name, seed)


class Sampler(object):
    def __init__(self,
                 env_name,
                 # env_kwargs, -- we don't have kwargs for tw-env
                 batch_size,
                 agent, #policy,
                 seed=None,
                 env=None):
        self.env_name = env_name
        # self.env_kwargs = env_kwargs - we don't have kwargs for tw-env
        self.batch_size = batch_size
        self.agent = agent # self.policy = policy
        self.seed = seed

        if env is None:
            env = gym.make(env_name) # env = gym.make(env_name, **env_kwargs) -- we don't have kwargs for tw-env
        self.env = env
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        self.env.close()
        self.closed = False

    def sample_async(self, *args, **kwargs):
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        return self.sample_async(*args, **kwargs)
