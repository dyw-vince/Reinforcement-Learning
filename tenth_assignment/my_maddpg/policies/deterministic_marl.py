import torch
from copy import deepcopy
from torch.nn import Module, ModuleDict
from .core import ActorNet_DDPG, CriticNet


class Independent_DDPG_Policy(Module):
    def __init__(self,
                 action_space,
                 n_agents,
                 actor_representation,
                 critic_representation,
                 actor_hidden_size,
                 critic_hidden_size,
                 normalize,
                 initialize=None,
                 activation=None,
                 activation_action=None,
                 device=None,
                 **kwargs):
        super(Independent_DDPG_Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.model_keys = kwargs['model_keys']
        self.actor_representation_info_shape = {
            key: actor_representation[key].output_shapes for key in self.model_keys}
        self.critic_representation_info_shape = {key: critic_representation[key].output_shapes
                                                 for key in self.model_keys}
        self.use_parameter_sharing = kwargs['use_parameter_sharing']

        self.actor_representation = actor_representation
        self.critic_representation = critic_representation
        self.target_actor_representation = deepcopy(self.actor_representation)
        self.target_critic_representation = deepcopy(
            self.critic_representation)

        self.actor, self.target_actor = ModuleDict(), ModuleDict()
        self.critic, self.target_critic = ModuleDict(), ModuleDict()
        for key in self.model_keys:
            dim_action = self.action_space[key].shape[-1]
            dim_actor_in, dim_actor_out, dim_critic_in = self._get_actor_critic_input(
                self.actor_representation[key].output_shapes['state'][0], dim_action,
                self.critic_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = ActorNet_DDPG(dim_actor_in, dim_actor_out, actor_hidden_size,
                                            normalize, initialize, activation, activation_action, device)
            self.critic[key] = CriticNet(
                dim_critic_in, critic_hidden_size, normalize, initialize, activation, device)
            self.target_actor[key] = deepcopy(self.actor[key])
            self.target_critic[key] = deepcopy(self.critic[key])

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
            parameters_critic[key] = list(self.critic_representation[key].parameters()) + list(
                self.critic[key].parameters())
        return parameters_critic

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        raise NotImplementedError

    def forward(self, observation,
                agent_ids=None, agent_key=None,
                rnn_hidden=None):
        rnn_hidden_new, actions = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            outputs = self.actor_representation[key](observation[key])
            actor_in = outputs['state']
            actions[key] = self.actor[key](actor_in)
        return rnn_hidden_new, actions

    def Qpolicy(self,
                observation,
                actions,
                agent_ids=None,
                agent_key=None,
                rnn_hidden=None):
        raise NotImplementedError

    def Qtarget(self,
                next_observation,
                next_actions,
                agent_ids=None,
                agent_key=None,
                rnn_hiddens=None):
        raise NotImplementedError

    def Atarget(self,
                next_observation,
                agent_ids=None,
                agent_key=None,
                rnn_hidden=None):
        raise NotImplementedError

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_representation.parameters(), self.target_actor_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_representation.parameters(), self.target_critic_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MADDPG_Policy(Independent_DDPG_Policy):
    def __init__(self,
                 action_space,
                 n_agents,
                 actor_representation,
                 critic_representation,
                 actor_hidden_size,
                 critic_hidden_size,
                 normalize=None,
                 initialize=None,
                 activation=None,
                 activation_action=None,
                 device=None,
                 **kwargs):
        super(MADDPG_Policy, self).__init__(action_space, n_agents, actor_representation, critic_representation,
                                            actor_hidden_size, critic_hidden_size,
                                            normalize, initialize, activation, activation_action, device, **kwargs)

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = dim_critic_rep
        return dim_actor_in, dim_actor_out, dim_critic_in

    def Qpolicy(self,
                joint_observation=None,
                joint_actions=None,
                agent_ids=None,
                agent_key=None,
                rnn_hidden=None):
        rnn_hidden_new, q_eval = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = 1

        critic_rep_in = torch.concat(
            [joint_observation, joint_actions], dim=-1)

        outputs = {k: self.critic_representation[k](
            critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            joint_rep_out = outputs[key]['state'].reshape(bs, -1)
            critic_in = joint_rep_out
            q_eval[key] = self.critic[key](critic_in)
        return rnn_hidden_new, q_eval

    def Qtarget(self,
                joint_observation=None,
                joint_actions=None,
                agent_ids=None,
                agent_key=None,
                rnn_hidden=None):
        rnn_hidden_new, q_target = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = 1

        critic_rep_in = torch.concat(
            [joint_observation, joint_actions], dim=-1)

        outputs = {k: self.target_critic_representation[k](
            critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            joint_rep_out = outputs[key]['state'].reshape(bs, -1)
            critic_in = joint_rep_out
            q_target[key] = self.target_critic[key](critic_in)
        return rnn_hidden_new, q_target

    def Atarget(self,
                next_observation,
                agent_ids=None,
                agent_key=None,
                rnn_hidden=None):
        rnn_hidden_new, next_actions = deepcopy(rnn_hidden), {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            outputs = self.target_actor_representation[key](
                next_observation[key])

            actor_in = outputs['state']
            next_actions[key] = self.target_actor[key](actor_in)
        return rnn_hidden_new, next_actions
