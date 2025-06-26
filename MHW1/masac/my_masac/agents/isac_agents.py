
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
            #    rnn_hidden = None,
               test_mode = False,
               **kwargs):
        """
        Returns actions for agents.

        Parameters:
            obs_dict (List[dict]): Observations for each agent in self.agent_keys.
            avail_actions_dict (Optional[List[dict]]): Actions mask values, default is None.
            rnn_hidden (Optional[dict]): The hidden variables of the RNN.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            rnn_hidden_state (dict): The new hidden states for RNN (if self.use_rnn=True).
            actions_dict (dict): The output actions.
        """
        batch_size = len(obs_dict)

        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict)
        hidden_state, actions, _ = self.policy(observation=obs_input, agent_ids=agents_id,
                                               avail_actions=avail_actions_input)#, rnn_hidden=rnn_hidden
        for key in self.agent_keys:
            if self.continuous_control:
                actions[key] = actions[key].reshape(batch_size, -1).cpu().detach().numpy()
            else:
                actions[key] = actions[key].reshape(batch_size).cpu().detach().numpy()
        actions_dict = [{k: actions[k][i] for k in self.agent_keys} for i in range(batch_size)]

        return {"hidden_state": hidden_state, "actions": actions_dict}
