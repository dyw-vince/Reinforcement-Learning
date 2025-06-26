from argparse import Namespace
# from my_masac.common import List, Optional, Union
from my_maddpg.agents.off_policy_marl import OffPolicyMARLAgents


class IDDPG_Agents(OffPolicyMARLAgents):
    """The implementation of Independent DDPG agents.

    Args:
        config: The Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """

    def __init__(self,
                 config: Namespace,
                 envs):
        super(IDDPG_Agents, self).__init__(config, envs)

        self.start_noise, self.end_noise = config.start_noise, config.end_noise
        self.noise_scale = config.start_noise
        self.delta_noise = (self.start_noise -
                            self.end_noise) / config.running_steps

        # build policy, optimizers, schedulers
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(
            self.config, self.model_keys, self.agent_keys, self.policy)

    def _build_policy(self):
        raise NotImplementedError

    def action(self,
               obs_dict,
               avail_actions_dict=None,
               test_mode=False,
               **kwargs):
        batch_size = len(obs_dict)

        obs_input, agents_id, _ = self._build_inputs(obs_dict)
        hidden_state, actions = self.policy(
            observation=obs_input, agent_ids=agents_id)

        for key in self.agent_keys:
            actions[key] = actions[key].reshape(
                batch_size, -1).cpu().detach().numpy()
        if not test_mode:
            actions = self.exploration(batch_size, actions)
        actions_dict = [{k: actions[k][i]
                         for k in self.agent_keys} for i in range(batch_size)]

        return {"hidden_state": hidden_state, "actions": actions_dict}
