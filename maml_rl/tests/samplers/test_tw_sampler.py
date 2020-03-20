import pytest

import numpy as np
import gym

import maml_rl.envs
from maml_rl.samplers import MultiTaskSampler
from maml_rl.episode import BatchEpisodes
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.envs.textworld import TextworldEnv, register_games

import textworld.gym
import textworld.gym.spaces

from typing import List, Optional

from textworld import EnvInfos

# @pytest.mark.parametrize('gamefile', ['tw-cooking.z8'])
# @pytest.mark.parametrize('num_workers', [1, 4])
# def test_init(gamefile, num_workers):
#     vocab = list(map(str, range(500)))
#     env_name = register_games(
#         [gamefile],
#     )

#     batch_size = 10
#     # Environment
#     env = gym.make(env_name)
#     env.close()
#     # Policy and Baseline
#     policy = get_policy_for_env(env)
#     baseline = LinearFeatureBaseline(get_input_size(env))
#     # baseline = LinearFeatureBaseline(10)

#     sampler = MultiTaskSampler(env_name,
#                                {}, # env_kwargs
#                                batch_size,
#                                policy,
#                                baseline,
#                                num_workers=num_workers)
#     sampler.close()


# @pytest.mark.parametrize('gamefile', ['tw-cooking.z8'])
# @pytest.mark.parametrize('batch_size', [1, 7])
# @pytest.mark.parametrize('num_tasks', [1, 5])
# @pytest.mark.parametrize('num_steps', [1, 3])
# @pytest.mark.parametrize('num_workers', [1, 3])
@pytest.mark.parametrize('gamefile', ['tw-cooking.z8'])
@pytest.mark.parametrize('batch_size', [1])
@pytest.mark.parametrize('num_tasks', [1])
@pytest.mark.parametrize('num_steps', [1])
@pytest.mark.parametrize('num_workers', [1])
def test_sample(gamefile, batch_size, num_tasks, num_steps, num_workers):
    request_infos = textworld.EnvInfos(feedback=True, facts=True, admissible_commands=True)
    env_name = register_games(
        [gamefile],
        request_infos=request_infos,
    )

    # Environment
    env = gym.make(env_name)
    env.close()
    # Policy and Baseline
    policy = get_policy_for_env(env)
    baseline = LinearFeatureBaseline(get_input_size(env))
    # baseline = LinearFeatureBaseline(10)

    sampler = MultiTaskSampler(env_name,
                               {}, # env_kwargs
                               batch_size,
                               policy,
                               baseline,
                               num_workers=num_workers)
    tasks = sampler.sample_tasks(num_tasks=num_tasks)
    train_episodes, valid_episodes = sampler.sample(tasks,
                                                    num_steps=num_steps)
    sampler.close()
    pytest.set_trace()

    assert len(train_episodes) == num_steps
    assert len(train_episodes[0]) == num_tasks
    assert isinstance(train_episodes[0][0], BatchEpisodes)

    assert len(valid_episodes) == num_tasks
    assert isinstance(valid_episodes[0], BatchEpisodes)
