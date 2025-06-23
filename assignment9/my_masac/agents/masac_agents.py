import torch
import torch.nn as nn
from argparse import Namespace
from my_masac.agents.isac_agents import ISAC_Agents
from my_masac.policies.gaussian_marl import MASAC_Policy

class MASAC_Agents(ISAC_Agents):
    def __init__(self,
                 config: Namespace,
                 envs):
        super(MASAC_Agents, self).__init__(config, envs)

    def _build_policy(self):
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = None
        initializer = torch.nn.init.orthogonal_
        activation = nn.LeakyReLU
        device = self.device
        agent = self.config.agent

        # build representations
        A_representation = self._build_representation(self.observation_space)
        critic_in = [sum(self.observation_space[k].shape) + sum(self.action_space[k].shape) for k in self.agent_keys]
        space_critic_in = {k: (sum(critic_in),) for k in self.agent_keys}
        C_representation = self._build_representation(space_critic_in)

        
        policy = MASAC_Policy(
            action_space=self.action_space, n_agents=self.n_agents,
            actor_representation=A_representation, critic_representation=C_representation,
            actor_hidden_size=self.config.actor_hidden_size,
            critic_hidden_size=self.config.critic_hidden_size,
            normalize=normalize_fn, initialize=initializer, activation=activation,
            activation_action= nn.Sigmoid,
            device=device, use_distributed_training=self.distributed_training,
            use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
            )
        

        return policy
