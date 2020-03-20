import gym
import torch
import json
import os
import yaml
from tqdm import trange

import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns

####
# TWR imports
import sys
sys.path.append("/data0/adadhika/ash_ppo_TWR/")
from ppo_agent import Agent
import generic
import evaluate as evaluate
import reinforcement_learning_dataset
from generic import HistoryScoreCache, EpisodicCountingMemory
from generic import to_pt

def main(args):

    ####
    # TWR
    with open(args.twconfig) as reader:
        twconfig = yaml.safe_load(reader)
    #twconfig = args.twconfig
    #twconfig = generic.load_config()
    agent = Agent(twconfig)
    # output_dir = os.getenv('PT_OUTPUT_DIR', '/tmp') if agent.philly else "."
    # data_dir = os.environ['PT_DATA_DIR'] if agent.philly else "."
    ####
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        config_filename = os.path.join(args.output_folder, 'config.json')

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    import pdb
    pdb.set_trace()
    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    # Policy
    # policy = get_policy_for_env(env,
    #                            hidden_sizes=config['hidden-sizes'],
    #                            nonlinearity=config['nonlinearity'])
    # --replaced by agent
    import pdb
    pdb.set_trace()
    agent.policy_net.share_memory() #policy.share_memory()
    pdb.set_trace()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env)) ## input_size can agent.block_hidden_dim
    pdb.set_trace()

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               agent=agent, # policy=policy
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)
    pdb.set_trace()
    metalearner = MAMLTRPO(agent, #policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device)

    num_iterations = 0
    for batch in trange(config['num-batches']):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        logs = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        logs.update(tasks=tasks,
                    num_iterations=num_iterations,
                    train_returns=get_returns(train_episodes[0]),
                    valid_returns=get_returns(valid_episodes))

        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str,
        help='name of the output folder')
    misc.add_argument('--seed', type=int, default=None,
        help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    # TW
    tw_args = parser.add_argument_group('TextWorld')
    tw_args.add_argument('--twconfig', type=str, required=True, help='parth to the textworld config file')
    tw_args.add_argument("-p", "--params", nargs="+", metavar="my.setting=value", default=[], help="override params of the config file, e.g. -p 'training.gamma=0.95'")

    args = parser.parse_args()

    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
