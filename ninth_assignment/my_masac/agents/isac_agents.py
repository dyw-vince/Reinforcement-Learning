
from argparse import Namespace
# from my_masac.common import List, Optional, Union
from my_masac.agents.off_policy_marl import OffPolicyMARLAgents


class ISAC_Agents(OffPolicyMARLAgents):
    """The implementation of Independent SAC agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs):#, SubprocVecMultiAgentEnv
        super(ISAC_Agents, self).__init__(config, envs)
        # build policy, optimizers, schedulers
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy)

    def _build_policy(self):
        raise NotImplementedError

    def action(self,
               obs_dict,
               avail_actions_dict = None,
               test_mode = False,
               **kwargs):
        batch_size = len(obs_dict)

        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict)
        hidden_state, actions, _ = self.policy(observation=obs_input, agent_ids=agents_id,
                                               avail_actions=avail_actions_input)#, rnn_hidden=rnn_hidden
        for key in self.agent_keys:
            actions[key] = actions[key].reshape(batch_size, -1).cpu().detach().numpy()
        actions_dict = [{k: actions[k][i] for k in self.agent_keys} for i in range(batch_size)]

        return {"hidden_state": hidden_state, "actions": actions_dict}
