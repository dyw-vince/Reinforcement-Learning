import numpy as np
from my_maddpg.utils.utils import space2shape, create_memory


class MARL_OffPolicyBuffer():
    """
    Replay buffer for off-policy MARL algorithms.

    """

    def __init__(self,
                 agent_keys,
                 state_space=None,
                 obs_space=None,
                 act_space=None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 batch_size: int = 1,
                 **kwargs):
        super(MARL_OffPolicyBuffer, self).__init__()
        self.agent_keys, self.state_space, self.obs_space, self.act_space, self.n_envs, self.buffer_size = agent_keys, state_space, obs_space, act_space, n_envs, buffer_size
        self.batch_size = batch_size
        self.store_global_state = False if self.state_space is None else True
        self.use_actions_mask = kwargs['use_actions_mask'] if 'use_actions_mask' in kwargs else False
        self.avail_actions_shape = kwargs['avail_actions_shape'] if 'avail_actions_shape' in kwargs else None
        assert self.buffer_size % self.n_envs == 0, "buffer_size must be divisible by the number of envs (parallels)"
        self.n_size = self.buffer_size // self.n_envs
        self.ptr = 0  # last data pointer
        self.size = 0  # current buffer size

        self.data = {}
        self.clear()
        self.data_keys = self.data.keys()

    def full(self):
        return self.size >= self.n_size

    def clear(self):
        reward_space = {key: () for key in self.agent_keys}
        terminal_space = {key: () for key in self.agent_keys}
        agent_mask_space = {key: () for key in self.agent_keys}

        self.data = {
            'obs': create_memory(space2shape(self.obs_space), self.n_envs, self.n_size),
            'actions': create_memory(space2shape(self.act_space), self.n_envs, self.n_size),
            'obs_next': create_memory(space2shape(self.obs_space), self.n_envs, self.n_size),
            'rewards': create_memory(reward_space, self.n_envs, self.n_size),
            'terminals': create_memory(terminal_space, self.n_envs, self.n_size, np.bool_),
            'agent_mask': create_memory(agent_mask_space, self.n_envs, self.n_size, np.bool_),
        }
        if self.store_global_state:
            self.data.update({
                'state': create_memory(space2shape(self.state_space), self.n_envs, self.n_size),
                'state_next': create_memory(space2shape(self.state_space), self.n_envs, self.n_size)
            })
        if self.use_actions_mask:
            self.data.update({
                "avail_actions": create_memory(self.avail_actions_shape, self.n_envs, self.n_size, np.bool_),
                "avail_actions_next": create_memory(self.avail_actions_shape, self.n_envs, self.n_size, np.bool_)
            })
        self.ptr, self.size = 0, 0

    def store(self, **step_data):
        """ Stores a step of data into the replay buffer. """
        for data_key, data_values in step_data.items():
            if data_key in ['state', 'state_next']:
                self.data[data_key][:, self.ptr] = data_values
                continue
            for agt_key in self.agent_keys:
                self.data[data_key][agt_key][:,
                                             self.ptr] = data_values[agt_key]
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = np.min([self.size + 1, self.n_size])

    def sample(self, batch_size=None):
        """
        Samples a batch of data from the replay buffer.

        Parameters:
            batch_size (int): The size of the batch data to be sampled.

        Returns:
            samples_dict (dict): The sampled data.
        """
        assert self.size > 0, "Not enough transitions for off-policy buffer to random sample."
        if batch_size is None:
            batch_size = self.batch_size
        env_choices = np.random.choice(self.n_envs, batch_size)
        step_choices = np.random.choice(self.size, batch_size)
        samples_dict = {}
        for data_key in self.data_keys:
            if data_key in ['state', 'state_next']:
                samples_dict[data_key] = self.data[data_key][env_choices, step_choices]
                continue
            samples_dict[data_key] = {
                k: self.data[data_key][k][env_choices, step_choices] for k in self.agent_keys}
        samples_dict['batch_size'] = batch_size
        return samples_dict

    def finish_path(self, *args, **kwargs):
        return
