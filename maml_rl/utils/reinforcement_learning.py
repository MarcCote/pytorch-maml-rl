import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from maml_rl.utils.torch_utils import weighted_mean, to_numpy

def value_iteration(transitions, rewards, gamma=0.95, theta=1e-5):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    delta = np.inf
    while delta >= theta:
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        new_values = np.max(q_values, axis=1)
        delta = np.max(np.abs(new_values - values))
        values = new_values

    return values

def value_iteration_finite_horizon(transitions, rewards, horizon=10, gamma=0.95):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    for k in range(horizon):
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        values = np.max(q_values, axis=1)

    return values

def get_returns(episodes):
    return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])

def reinforce_loss(agent, episodes, params=None): # need to incorporate `params` in ppo_model.py
    obs_str = episodes.observations
    triplets = episodes.triplets
    acl = episodes.candidates
    ads = episodes.advantages
    chosen_indices = episodes.chosen_indices
    eps_lengths = episodes.lengths
    max_ep_length = len(episodes)
    '''print("chosen indices")
    print(len(chosen_indices))
    print(len(chosen_indices[0]))
    print("-------")
    print(chosen_indices[-1])
    print("....")
    print(chosen_indices[0])
    print(chosen_indices[0][0])'''
    '''print("REINFORCE OBS")
    print(len(obs_str))
    print(len(obs_str[0]))
    print(obs_str[0])
    print(len(obs_str[0][0]))
    print("hi")
    print(obs_str[0][0])
    print("REINFORCE ADS")
    print(ads.shape)
    print(ads)
    print(type(ads))
    print(type(ads[0]))
    print(len(ads))
    print(len(ads[0]))
    print(ads[0])
    print('hi')
    print(episodes.batch_size)'''
    log_probs = []

    if params is not None:
        old_model_dict = agent.policy_net.state_dict()
        model_dict = deepcopy(old_model_dict)
        inner_loop_params = {k: v for k,v in params.items() if k in model_dict}
        model_dict.update(inner_loop_params)
        agent.policy_net.load_state_dict(model_dict)
    #print("In RF")
    #print(episodes.batch_size)
    #print(len(episodes))
    for i in range(episodes.batch_size):
        h_og, obs_mask, h_go, node_mask = agent.encode(obs_str[i], triplets[i], use_model='policy')
        #print('hgo_shape =  ' + str(h_go.shape))
        action_features, value, action_masks, new_h, new_c = agent.action_scoring(acl[i], h_og, obs_mask, h_go, node_mask, use_model='policy')
        #print("AF")
        #print(action_features.shape)
        #print(chosen_indices[i])
        pi = agent.policy_net.dist(probs=action_features)
        for j in range(len(chosen_indices[i])):
            if len(chosen_indices[i][j].shape)==1:
                continue
            chosen_indices[i][j] = chosen_indices[i][j].unsqueeze(0)
        ci = torch.cat(chosen_indices[i])
        #print('CI shape ' + str(ci.shape))
        unpadded_log_probs = pi.log_prob(torch.cat(chosen_indices[i]))
        #print('unpadded_log_provs shape '+ str(unpadded_log_probs.shape))
        padded_log_probs = F.pad(unpadded_log_probs, (0, max_ep_length - eps_lengths[i])).reshape(len(episodes), -1)
        #print('padded_log_prbs shape ' + str(padded_log_probs.shape))
        log_probs.append(padded_log_probs)
        #print('ULP-shape : ' + str(unpadded_log_probs.shape))
        #log_probs.append(pi.log_prob(torch.cat(chosen_indices[i])).reshape(len(episodes), -1))
    #print('log_probs shape : ' + str(len(log_probs)))
    #print([elem.shape for elem in log_probs])
    torch_log_probs = torch.cat(log_probs, 0).reshape(len(episodes), episodes.batch_size)
    #print(torch_log_probs.shape)
    
    ## get obs, triplets, acl, meta_dones, step _ rewards
    #h_og, obs_mask, h_go, node_mask = agent.encode(observation_str, triplets, use_model="policy")
    #action_features, value, action_masks, new_h, new_c = agent.action_scoring(action_candidate_list, step_rewards, h_og, obs_mask, h_go, node_mask, previous_h, previous_c, use_model="policy")
    #pi = agent.policy_net.dist(logits=action_features) # make sure it's logits/probs..
    #pi = policy(episodes.observations.view((-1, *episodes.observation_shape)),
    #            params=params)

    #log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
    #log_probs = log_probs.view(len(episodes), episodes.batch_size)

    losses = -weighted_mean(torch_log_probs * episodes.advantages,
                            lengths=episodes.lengths)
    if params is not None:  # RELOAD
        agent.policy_net.load_state_dict(old_model_dict)
    #print("about to exit reinforce loss : " + str(losses))
    return losses.mean()
