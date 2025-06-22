import torch
from copy import deepcopy
from torch.nn import Module,ModuleDict
from .core import ActorNet_SAC, CriticNet

class Basic_ISAC_Policy(Module):

    def __init__(self,
                 action_space,
                 n_agents,
                 actor_representation,
                 critic_representation,
                 actor_hidden_size,
                 critic_hidden_size,
                 normalize,
                 initialize = None,
                 activation = None,
                 activation_action = None,
                 device = None,
                 **kwargs):
        super(Basic_ISAC_Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']

        self.actor_representation = actor_representation
        self.critic_1_representation = critic_representation
        self.critic_2_representation = deepcopy(critic_representation)
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)

        self.actor, self.critic_1, self.critic_2 = ModuleDict(), ModuleDict(), ModuleDict()
        for key in self.model_keys:
            dim_action = self.action_space[key].shape[-1]
            dim_actor_in, dim_actor_out, dim_critic_in = self._get_actor_critic_input(
                self.actor_representation[key].output_shapes['state'][0], dim_action,
                self.critic_1_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = ActorNet_SAC(dim_actor_in, dim_actor_out, actor_hidden_size,
                                                   normalize, initialize, activation, activation_action, device)
            self.critic_1[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation, device)
            self.critic_2[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation, device)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

        # Prepare DDP module.
        
        
    @property
    def parameters_actor(self):
        parameters_actor = {}
        for key in self.model_keys:
            parameters_actor[key] = list(self.actor_representation[key].parameters()) + list(
                self.actor[key].parameters())
        return parameters_actor

    @property
    def parameters_critic(self):
        parameters_critic = {}
        for key in self.model_keys:
            parameters_critic[key] = list(self.critic_1_representation[key].parameters()) + list(
                self.critic_1[key].parameters()) + list(self.critic_2_representation[key].parameters()) + list(
                self.critic_2[key].parameters())
        return parameters_critic

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        raise NotImplementedError


    def forward(self, observation, agent_ids = None,
                avail_actions = None, agent_key = None,
                rnn_hidden = None):
        rnn_hidden_new, act_dists, actions_dict, log_action_prob = deepcopy(rnn_hidden), {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            outputs = self.actor_representation[key](observation[key])
            actor_in = outputs['state']
            act_dists = self.actor[key](actor_in)
            actions_dict[key], log_action_prob[key] = act_dists.activated_rsample_and_logprob()
        return rnn_hidden_new, actions_dict, log_action_prob

    # def Qpolicy(self, observation,
    #             actions,
    #             agent_ids = None, agent_key = None,
    #             rnn_hidden_critic_1 = None,
    #             rnn_hidden_critic_2 = None):
    #     raise NotImplementedError


    # def Qtarget(self, next_observation,
    #             next_actions,
    #             agent_ids = None, agent_key = None,
    #             rnn_hidden_critic_1 = None,
    #             rnn_hidden_critic_2 = None):
    #     raise NotImplementedError


    # def Qaction(self, observation,
    #             actions,
    #             agent_ids, agent_key = None,
    #             rnn_hidden_critic_1 = None,
    #             rnn_hidden_critic_2 = None):
    #     raise NotImplementedError


    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1_representation.parameters(), self.target_critic_1_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2_representation.parameters(), self.target_critic_2_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MASAC_Policy(Basic_ISAC_Policy):

    def __init__(self,
                 action_space,
                 n_agents,
                 actor_representation,
                 critic_representation,
                 actor_hidden_size,
                 critic_hidden_size,
                 normalize = None,
                 initialize = None,
                 activation = None,
                 activation_action = None,
                 device = None,
                 **kwargs):
        super(MASAC_Policy, self).__init__(action_space, n_agents, actor_representation, critic_representation,
                                           actor_hidden_size, critic_hidden_size,
                                           normalize, initialize, activation, activation_action, device,
                                            **kwargs)

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = dim_critic_rep
        return dim_actor_in, dim_actor_out, dim_critic_in

    def Qpolicy(self, joint_observation = None,
                joint_actions = None,
                agent_ids = None, agent_key = None,
                rnn_hidden_critic_1 = None,
                rnn_hidden_critic_2 = None):
        
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = 1

        critic_rep_in = torch.concat([joint_observation, joint_actions], dim=-1)
        
        outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in) for k in agent_list}
        outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, -1)
            joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, -1)
            critic_1_in = joint_rep_out_1
            critic_2_in = joint_rep_out_2
            q_1[key] = self.critic_1[key](critic_1_in)
            q_2[key] = self.critic_2[key](critic_2_in)

        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, q_1, q_2

    def Qtarget(self, joint_observation= None,
                joint_actions = None,
                agent_ids = None, agent_key = None,
                rnn_hidden_critic_1 = None,
                rnn_hidden_critic_2 = None):
        
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        target_q = {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = 1
        critic_rep_in = torch.concat([joint_observation, joint_actions], dim=-1)
        outputs_critic_1 = {k: self.target_critic_1_representation[k](critic_rep_in) for k in agent_list}
        outputs_critic_2 = {k: self.target_critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, -1)
            joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, -1)
            critic_1_in = joint_rep_out_1
            critic_2_in = joint_rep_out_2
            q_1 = self.target_critic_1[key](critic_1_in)
            q_2 = self.target_critic_2[key](critic_2_in)
            target_q[key] = torch.min(q_1, q_2)
        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, target_q

    def Qaction(self, joint_observation = None,
                joint_actions = None,
                agent_ids = None, agent_key = None,
                rnn_hidden_critic_1 = None,
                rnn_hidden_critic_2 = None):
        
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len =  1

        critic_rep_in = torch.concat([joint_observation, joint_actions], dim=-1)
        outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in) for k in agent_list}
        outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
           
            joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, -1)
            joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, -1)
            critic_1_in = joint_rep_out_1
            critic_2_in = joint_rep_out_2

            q_1[key] = self.critic_1[key](critic_1_in)
            q_2[key] = self.critic_2[key](critic_2_in)

        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, q_1, q_2
