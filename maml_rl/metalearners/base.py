import torch
import asyncio

from maml_rl.samplers import MultiTaskSampler


class GradientBasedMetaLearner(object):
    def __init__(self, agent, device='cpu'): # policy, device='cpu'):
        self.device = torch.device(device)
        self.agent = agent #self.policy = policy
        self.agent.policy_net.to(self.device) #self.policy.to(self.device)
        self._event_loop = asyncio.get_event_loop()

    def adapt(self, episodes, *args, **kwargs):
        raise NotImplementedError()

    def step(self, train_episodes, valid_episodes, *args, **kwargs):
        raise NotImplementedError()

    def _async_gather(self, coroutines):
        coroutine = asyncio.gather(*coroutines)
        return zip(*self._event_loop.run_until_complete(coroutine))
