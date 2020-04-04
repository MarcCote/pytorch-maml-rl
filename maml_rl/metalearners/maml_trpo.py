from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from maml_rl.samplers import MultiTaskSampler
from maml_rl.metalearners.base import GradientBasedMetaLearner
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters)
from maml_rl.utils.optimization import conjugate_gradient
from maml_rl.utils.reinforcement_learning import reinforce_loss


class MAMLTRPO(GradientBasedMetaLearner):
    """Model-Agnostic Meta-Learning (MAML, [1]) for Reinforcement Learning
    application, with an outer-loop optimization based on TRPO [2].

    Parameters
    ----------
    policy : `maml_rl.policies.Policy` instance
        The policy network to be optimized. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    fast_lr : float
        Step-size for the inner loop update/fast adaptation.

    num_steps : int
        Number of gradient steps for the fast adaptation. Currently setting
        `num_steps > 1` does not resample different trajectories after each
        gradient steps, and uses the trajectories sampled from the initial
        policy (before adaptation) to compute the loss at each step.

    first_order : bool
        If `True`, then the first order approximation of MAML is applied.

    device : str ("cpu" or "cuda")
        Name of the device for the optimization.

    References
    ----------
    .. [1] Finn, C., Abbeel, P., and Levine, S. (2017). Model-Agnostic
           Meta-Learning for Fast Adaptation of Deep Networks. International
           Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., and Abbeel, P.
           (2015). Trust Region Policy Optimization. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self,
                 agent, #policy,
                 fast_lr=0.5,
                 first_order=False,
                 device='cpu'):
        super(MAMLTRPO, self).__init__(agent, device=device) # policy, device=device)
        self.fast_lr = fast_lr
        self.first_order = first_order

    async def adapt(self, train_futures, first_order=None):
        if first_order is None:
            first_order = self.first_order
        # Loop over the number of steps of adaptation
        params = None
        for futures in train_futures:
            inner_loss = reinforce_loss(self.agent, #self.policy,
                                        await futures,
                                        params=params)
            params = self.agent.policy_net.update_params(inner_loss, #self.policy.update_params(inner_loss,
                                               params=params,
                                               step_size=self.fast_lr,
                                               first_order=first_order)
        return params

    def hessian_vector_product(self, kl, damping=1e-2):
        grads = torch.autograd.grad(kl,
                                    self.agent.policy_net.parameters(), # self.policy.parameters(),
                                    create_graph=True, allow_unused=True)
        non_none_grads = []
        for elem in grads:
             if elem is not None:
                 non_none_grads.append(elem.contiguous())
        grads = tuple(non_none_grads)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                         self.agent.policy_net.parameters(), #self.policy.parameters(),
                                         retain_graph=retain_graph, allow_unused=True)
            non_none_grad2s = []
            for elem in grad2s:
                if elem is not None:
                    non_none_grad2s.append(elem.contiguous())
            grad2s = tuple(non_none_grad2s)
            #print('grad2s')
            #print(grad2s)

            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    async def surrogate_loss(self, train_futures, valid_futures, old_pi=None):
        first_order = (old_pi is not None) or self.first_order
        params = await self.adapt(train_futures,
                                  first_order=first_order)

        with torch.set_grad_enabled(old_pi is None):
            valid_episodes = await valid_futures
            if params is not None: # can make a function
                model_dict = self.agent.policy_net.state_dict()
                old_model_dict = deepcopy(model_dict)
                inner_loop_params = {k: v for k, v in params.items() if k in model_dict}
                model_dict.update(inner_loop_params)
                self.agent.policy_net.load_state_dict(model_dict)

            ######
            ## can go in a function in ppo_agent.py
            valid_obs_str = valid_episodes.observations
            valid_triplets = valid_episodes.triplets
            valid_acl = valid_episodes.candidates
            valid_ads = valid_episodes.advantages
            valid_chosen_indices = valid_episodes.chosen_indices
            valid_episode_lengths = valid_episodes.lengths
            valid_max_eps_length = len(valid_episodes)

            #old_log_probs, new_log_probs = [], []
            ratios, kls, old_pis_ = [], [], []
            for i in range(valid_episodes.batch_size):
                h_og, obs_mask, h_go, node_mask = self.agent.encode(valid_obs_str[i], valid_triplets[i], use_model='policy')
                action_features, value, action_masks, new_h, new_c = self.agent.action_scoring(valid_acl[i], h_og, obs_mask, h_go, node_mask, use_model='policy')
                #print('action_features.shape : ')
                #print(action_features.shape)
                #print('AF')
                #print(action_features)

                pi = self.agent.policy_net.dist(probs=action_features)
                #print("OLD pi : ")
                #print(old_pi)
                if old_pi is None:
                    old_pi_ = detach_distribution(pi)
                else:
                    old_pi_ = old_pi[i]

                for j in range(len(valid_chosen_indices[i])):
                    if len(valid_chosen_indices[i][j].shape)==1:
                        continue
                    valid_chosen_indices[i][j] = valid_chosen_indices[i][j].unsqueeze(0)

                #new_log_prob = pi.log_prob(torch.cat(valid_chosen_indices[i])).reshape(len(valid_episodes), -1)
                #old_log_prob = old_pi_.log_prob(torch.cat(valid_chosen_indices[i])).reshape(len(valid_episodes), -1)

                unpadded_new_log_prob = pi.log_prob(torch.cat(valid_chosen_indices[i]))
                unpadded_old_log_prob = old_pi_.log_prob(torch.cat(valid_chosen_indices[i]))

                padded_new_log_prob = F.pad(unpadded_new_log_prob, (0, valid_max_eps_length-valid_episode_lengths[i])).reshape(len(valid_episodes), -1)
                padded_old_log_prob = F.pad(unpadded_old_log_prob, (0, valid_max_eps_length-valid_episode_lengths[i])).reshape(len(valid_episodes), -1)

                #print(kl_divergence(pi, old_pi_))
                #print(kl_divergence(pi, old_pi_).shape)
                log_ratio = padded_new_log_prob - padded_old_log_prob
                ratios.append(torch.exp(log_ratio))
                kls.append(F.pad(kl_divergence(pi, old_pi_), (0, valid_max_eps_length-valid_episode_lengths[i])))
                old_pis_.append(old_pi_)
                #print('KLS')
                #print(kls.shape)

            #print("KLS")
            #print(kls)
            #print(type(kls[0]))
            #print(kls[0].shape)
            torch_kls = torch.cat(kls, 0).reshape(len(valid_episodes), valid_episodes.batch_size)

            torch_ratios = torch.cat(ratios, 0).reshape(len(valid_episodes), valid_episodes.batch_size)
            losses = -weighted_mean(torch_ratios * valid_episodes.advantages, lengths=valid_episodes.lengths)

            kls = weighted_mean(torch_kls, lengths=valid_episodes.lengths)
            old_pis = old_pis_
            if params is not None:
                self.agent.policy_net.load_state_dict(old_model_dict) # Reload

            '''#return losses.mean(), kls.mean(), old_pi
                

            pi = torch.cat(pi, 0) 
            print(pi.shape)
                
            #pi = self.policy(valid_episodes.observations, params=params)
            #######
            if old_pi is None:
                old_pi = detach_distribution(pi)

            log_ratio = (pi.log_prob(valid_episodes.actions)
                         - old_pi.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)

            losses = -weighted_mean(ratio * valid_episodes.advantages,
                                    lengths=valid_episodes.lengths)
            kls = weighted_mean(kl_divergence(pi, old_pi),
                                lengths=valid_episodes.lengths)'''

        return losses.mean(), kls.mean(), old_pis

    def step(self,
             train_futures,
             valid_futures,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        num_tasks = len(train_futures[0])
        logs = {}

        # Compute the surrogate loss
        old_losses, old_kls, old_pis = self._async_gather([
            self.surrogate_loss(train, valid, old_pi=None)
            for (train, valid) in zip(zip(*train_futures), valid_futures)])

        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)

        old_loss = sum(old_losses) / num_tasks
        grads = torch.autograd.grad(old_loss,
                                    self.agent.policy_net.parameters(),
                                    retain_graph=True, allow_unused=True)
        non_none_grads = []
        non_none_indices = []
        for i in range(len(grads)):
            elem = grads[i]
            if elem is not None:
                non_none_indices.append(i)
                non_none_grads.append(elem.contiguous())
        grads = tuple(non_none_grads)
        #print("Type of grads")
        #print(grads)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir,
                              hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        non_none_params = []
        for i, param in enumerate(self.agent.policy_net.parameters()):
            if i in non_none_indices:
                non_none_params.append(param.contiguous())

        # Save the old parameters
        old_params = parameters_to_vector(non_none_params)
        #print("OLD PARAMS shape")
        #print(old_params.shape)
        #print(len(step))
        #print(len(non_none_indices))
        #print(type(self.agent.policy_net.parameters()))

        # Line search
        step_size = 1.0
        print("OLD PIS")
        print(old_pis)
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.agent.policy_net.parameters(), indices=non_none_indices) #self.agent.policy_net.parameters())

            losses, kls, _ = self._async_gather([
                self.surrogate_loss(train, valid, old_pi=old_pi)
                for (train, valid, old_pi)
                in zip(zip(*train_futures), valid_futures, old_pis)])

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.agent.policy_net.parameters(), indices=non_none_indices)

        return logs
