import torch
import torch.nn as nn
import torch.nn.functional as F
from debug import ForkedPdb as fpdb
from collections import OrderedDict

from layers import masked_mean

class LinearFeatureBaseline(nn.Module):
    """Linear baseline based on handcrafted features, as described in [1] 
    (Supplementary Material 2).

    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel, 
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016 
        (https://arxiv.org/abs/1604.06778)
    """
    def __init__(self, input_size, agent, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        self.agent = agent

        self.weight = nn.Parameter(torch.Tensor(self.feature_size,),
                                   requires_grad=False)
        self.weight.data.zero_()
        self._eye = torch.eye(self.feature_size,
                              dtype=torch.float32,
                              device=self.weight.device)

    '''@property
    def feature_size(self):
        return 2 * self.input_size + 4'''
    @property
    def feature_size(self):
        '''if self.agent.policy_net.enable_text_input==False:
            return self.agent.policy_net.block_hidden_dim * 99
        elif self.agent.policy_net.enable_graph_input==False:
            return self.agent.policy_net.block_'''
        return self.agent.policy_net.block_hidden_dim

    def _feature(self, episodes):
        ones = episodes.mask.unsqueeze(2)
        observations = episodes.observations
        time_step = torch.arange(len(episodes)).view(-1, 1, 1) * ones / 100.0

        return torch.cat([
            observations,
            observations ** 2,
            time_step,
            time_step ** 2,
            time_step ** 3,
            ones
        ], dim=2)

    def _tw_feature(self, episodes):
        ep_obs_str = episodes.observations
        ep_triplets = episodes.triplets
        ep_h_og = []
        ep_obs_mask = []
        ep_h_go = []
        ep_node_mask = []
        for i in range(episodes.batch_size):
            h_og, obs_mask, h_go, node_mask = self.agent.encode(ep_obs_str[i], ep_triplets[i], use_model="policy")
            #ep_h_og.append(h_og.unsqueeze(0))
            ep_h_go.append(h_go.unsqueeze(0))
            #ep_obs_mask.append(obs_mask.unsqueeze(0))
            ep_node_mask.append(node_mask.unsqueeze(0))
        print("in twfeature")
        print(len(ep_h_go))
        print(len(ep_h_go[0]))
        print(h_go.shape)
        print(torch.cat(ep_h_go, 0).shape)
        if self.agent.policy_net.enable_text_input==False:
            return torch.cat(ep_h_go, 0), torch.cat(ep_node_mask, 0)
        elif self.agent.policy_net.enable_graph_input==False:
            return torch.cat(ep_h_og, 0), torch.cat(ep_obs_mask, 0)
        else:
            return torch.cat(ep_h_og, 0), torch.cat(ep_h_go, 0), torch.cat(ep_obs_mask, 0), torch.cat(ep_node_mask, 0)
            

    def fit(self, episodes):
        # sequence_length * batch_size x feature_size
        # featmat = self._feature(episodes).view(-1, self.feature_size)
        # sequence_length * batch_size x 1
        ep_h_og, ep_node_masks = self._tw_feature(episodes) # currently for graph only setting
        masked_ep_h_go = ep_node_masks.unsqueeze(-1).expand(ep_h_og.shape) * ep_h_og
        agg_ep_h_go = masked_ep_h_go.view(masked_ep_h_go.shape[0] * masked_ep_h_go.shape[1], masked_ep_h_go.shape[-2], masked_ep_h_go.shape[-1])
        agg_ep_node_masks = ep_node_masks.view(ep_node_masks.shape[0] * ep_node_masks.shape[1], ep_node_masks.shape[-1])
        agg_ep_h_go = masked_mean(agg_ep_h_go, agg_ep_node_masks) # bs * seq_len X block_hidden_dim
        #print(ep_h_og[0].shape)
        #print(ep_h_og[1].shape)
        #print("----------------")
        #print(len(episodes.observations))
        #print(len(episodes.observations[0]))
        #print(len(episodes.observations[0][0]))
        #print("PRINTING CANDS")
        #print(len(episodes.candidates))
        #print(episodes.candidates[0])
        #print(episodes.candidates[0][0])
        #print(len(episodes.candidates[0]))
        #print(len(episodes.candidates[0][0]))
        #print("returns") 
        #print(episodes.returns)
        #print(len(episodes.returns))
        #print(len(episodes.returns[0]))
        returns = episodes.returns.view(-1, 1)
        print("RETRNS" + str(returns.shape))
        reg_coeff = self._reg_coeff
        #flattened_masked_ep_h_og = masked_ep_h_og.view(masked_ep_h_og.shape[0], -1)
        #XT_y = torch.matmul(featmat.t(), returns)
        XT_y = torch.matmul(agg_ep_h_go.permute(1, 0), returns)
        #XT_X = torch.matmul(featmat.t(), featmat)
        XT_X = torch.matmul(agg_ep_h_go.permute(1, 0), agg_ep_h_go)
        print(XT_y.shape)
        print(XT_X.shape)
        for _ in range(5):
            try:
                coeffs, _ = torch.lstsq(XT_y, XT_X + reg_coeff * self._eye)
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
            raise RuntimeError('Unable to solve the normal equations in '
                '`LinearFeatureBaseline`. The matrix X^T*X (with X the design '
                'matrix) is not full-rank, regardless of the regularization '
                '(maximum regularization: {0}).'.format(reg_coeff))
        self.weight.copy_(coeffs.flatten())

    def forward(self, episodes):
        features, masks = self._tw_feature(episodes)
        masked_features = masks.unsqueeze(-1).expand(features.shape) * features ## assuming graph only : bs X seq_length X num_nodes X block_dim
        agg_masked_features = masked_features.view(masked_features.shape[0] * masked_features.shape[1], masked_features.shape[-2], masked_features.shape[-1])
        agg_node_masks = masks.view(masks.shape[0] * masks.shape[1], masks.shape[-1])
        agg_masked_features = masked_mean(agg_masked_features, agg_node_masks) # bs * seq_len X block_hidden_dim
        values = torch.mv(agg_masked_features, self.weight)
        
        print("baseline for")
        print(values.shape)
        print(self.weight.shape)
        return values.view(features.shape[:2])
