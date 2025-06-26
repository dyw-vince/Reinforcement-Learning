import torch
import torch.nn as nn
from argparse import Namespace
from my_maddpg.environment import DummyVecMultiAgentEnv  # , SubprocVecMultiAgentEnv
from my_maddpg.agents.iddpg_agents import IDDPG_Agents
from my_maddpg.policies.deterministic_marl import MADDPG_Policy


class MADDPG_Agents(IDDPG_Agents):
    """The implementation of MASAC agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs):
        super(MADDPG_Agents, self).__init__(config, envs)

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

        # build representations
        A_representation = self._build_representation(
            self.config.representation, self.observation_space, self.config)
        critic_in = [sum(self.observation_space[k].shape) +
                     sum(self.action_space[k].shape) for k in self.agent_keys]
        space_critic_in = {k: (sum(critic_in), ) for k in self.agent_keys}
        C_representation = self._build_representation(
            self.config.representation, space_critic_in, self.config)

        # build policies
        policy = MADDPG_Policy(
            action_space=self.action_space, n_agents=self.n_agents,
            actor_representation=A_representation, critic_representation=C_representation,
            actor_hidden_size=self.config.actor_hidden_size,
            critic_hidden_size=self.config.critic_hidden_size,
            normalize=normalize_fn, initialize=initializer, activation=activation,
            activation_action=nn.Sigmoid,
            device=device,
            use_parameter_sharing=self.use_parameter_sharing,
            model_keys=self.model_keys,
        )
        self.continuous_control = True

        return policy
