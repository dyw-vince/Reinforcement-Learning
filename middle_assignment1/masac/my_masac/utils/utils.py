import os
import time
import random
import torch
import numpy as np
from typing import Dict
from gym import spaces
import scipy.signal

EPS = 1e-8


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def combine_actions(group_of_policy_out: list, n_envs: int):
    actions_groups = [group['actions'] for group in group_of_policy_out]
    actions_combined = [{} for _ in range(n_envs)]
    for i_env in range(n_envs):
        for actions in actions_groups:
            actions_combined[i_env].update(actions[i_env])
    return actions_combined


def space2shape(observation_space):
    """Convert gym.space variable to shape
    Args:
        observation_space: the space variable with type of gym.Space.

    Returns:
        The shape of the observation_space.
    """
    if isinstance(observation_space, Dict) or isinstance(observation_space, dict):
        return {key: observation_space[key].shape for key in observation_space.keys()}
    elif isinstance(observation_space, tuple):
        return observation_space
    else:
        return observation_space.shape


class EnvWrapper:

    def __init__(self, env, **kwargs):
        super(EnvWrapper, self).__init__()
        self.env = env
        self._action_space = None
        self._observation_space = None
        self._reward_range = None
        self._metadata = None
        self._max_episode_steps = None
        self._episode_step = 0
        self._episode_score = 0.0

    @property
    def action_space(self):
        """Returns the action space of the environment."""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space):
        """Sets the action space"""
        self._action_space = space

    @property
    def observation_space(self) -> spaces.Space:
        """Returns the observation space of the environment."""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space):
        """Sets the observation space."""
        self._observation_space = space

    @property
    def reward_range(self):
        """Return the reward range of the environment."""
        if self._reward_range is None:
            return self.env.reward_range
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value):
        """Sets reward range."""
        self._reward_range = value

    @property
    def metadata(self) -> dict:
        """Returns the environment metadata."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """Sets metadata"""
        self._metadata = value

    @property
    def max_episode_steps(self) -> int:
        """Returns the maximum of episode steps."""
        if self._max_episode_steps is None:
            return self.env.max_episode_steps
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, value):
        """Sets the maximum of episode steps"""
        self._max_episode_steps = value

    @property
    def render_mode(self):
        """Returns the environment render_mode."""
        return self.env.render_mode

    def step(self, action):
        """Steps through the environment with action."""
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with kwargs."""
        try:
            obs, info = self.env.reset(**kwargs)
        except:
            obs = self.env.reset(**kwargs)
            info = {}
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        return obs, info

    def render(self, *args, **kwargs):
        """Renders the environment."""
        return self.env.render(*args, **kwargs)

    def close(self):
        """Closes the environment."""
        return self.env.close()

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper."""
        return self.env


class MultiAgentEnvWrapper(EnvWrapper):

    def __init__(self, env, **kwargs):
        super(MultiAgentEnvWrapper, self).__init__(env, **kwargs)
        self._env_info = None
        self._state_space = None
        self.agents = self.env.agents
        self.num_agents = self.env.num_agents
        self.agent_groups = self.env.agent_groups
        self._episode_score = {agent: 0.0 for agent in self.agents}
        self.env_info = self.env.get_env_info()
        self.groups_info = self.env.get_groups_info()

    def reset(self, **kwargs):
        """Resets the environment with kwargs."""
        obs, info = self.env.reset(**kwargs)
        self._episode_step = 0
        self._episode_score = {agent: 0.0 for agent in self.agents}
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        info["agent_mask"] = self.agent_mask
        info["avail_actions"] = self.avail_actions
        info["state"] = self.state
        return obs, info

    def step(self, action):
        """Steps through the environment with action."""
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        self._episode_step += 1
        for agent in self.agents:
            self._episode_score[agent] += reward[agent]
        info["episode_step"] = self._episode_step  # current episode step
        info["episode_score"] = self._episode_score  # the accumulated rewards
        info["agent_mask"] = self.agent_mask
        info["avail_actions"] = self.avail_actions
        info["state"] = self.state
        return observation, reward, terminated, truncated, info

    @property
    def env_info(self):
        """Returns the information of the environment."""
        if self._env_info is None:
            return self.env.env_info
        return self._env_info

    @env_info.setter
    def env_info(self, info):
        """Sets the action space"""
        self._env_info = info

    @property
    def state_space(self) -> spaces.Space:
        """Returns the global state space of the environment."""
        if self._state_space is None:
            return self.env.state_space
        return self._state_space

    @state_space.setter
    def state_space(self, space: spaces.Space):
        """Sets the global state space."""
        self._state_space = space

    @property
    def state(self):
        """Returns global states in the multi-agent environment."""
        return self.env.state()

    @property
    def agent_mask(self):
        """Returns mask variables to mark alive agents in multi-agent environment."""
        return self.env.agent_mask()

    @property
    def avail_actions(self):
        """Returns mask variables to mark available actions for each agent."""
        return self.env.avail_actions()


def create_directory(path):
    """Create an empty directory.
    Args:
        path: the path of the directory
    """
    dir_split = path.split("/")
    current_dir = dir_split[0] + "/"
    for i in range(1, len(dir_split)):
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = current_dir + dir_split[i] + "/"


def get_time_string():
    t_now = time.localtime(time.time())
    t_year = str(t_now.tm_year).zfill(4)
    t_month = str(t_now.tm_mon).zfill(2)
    t_day = str(t_now.tm_mday).zfill(2)
    t_hour = str(t_now.tm_hour).zfill(2)
    t_min = str(t_now.tm_min).zfill(2)
    t_sec = str(t_now.tm_sec).zfill(2)
    time_string = f"{t_year}_{t_month}{t_day}_{t_hour}{t_min}{t_sec}"
    return time_string


def create_memory(shape,
                  n_envs,
                  n_size,
                  dtype=np.float32):
    """
    Create a numpy array for memory data.

    Args:
        shape: data shape.
        n_envs: number of parallel environments.
        n_size: length of data sequence for each environment.
        dtype: numpy data type.

    Returns:
        An empty memory space to store data. (initial: numpy.zeros())
    """
    if shape is None:
        return None
    elif isinstance(shape, dict):
        memory = {}
        for key, value in shape.items():
            if value is None:  # save an object type
                memory[key] = np.zeros([n_envs, n_size], dtype=object)
            else:
                memory[key] = np.zeros(
                    [n_envs, n_size] + list(value), dtype=dtype)
        return memory
    elif isinstance(shape, tuple):
        return np.zeros([n_envs, n_size] + list(shape), dtype)
    else:
        raise NotImplementedError


def combined_shape(length: int, shape=None):
    """Expand the original shape.

    Args:
        length (int): The length of the first dimension to prepend.
        shape (int, list, tuple, or None): The target shape to be expanded.
                                           It can be an integer, a sequence, or None.

    Returns:
        tuple: A new shape expanded from the input shape.

    Examples
    --------
        >>> length = 2
        >>> shape_1 = None
        >>> shape_2 = 3
        >>> shape_3 = [4, 5]
        >>> combined(length, shape_1)
        (2, )
        >>> combined(length, shape_2)
        (2, 3)
        >>> combined(length, shape_3)
        (2, 4, 5)
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount=0.99):
    """Get a discounted cumulated summation.
    Args:
        x: The original sequence. In DRL, x can be reward sequence.
        discount: the discount factor (gamma), default is 0.99.

    Returns:
        The discounted cumulative returns for each step.

    Examples
    --------
    >>> x = [0, 1, 2, 2]
    >>> y = discount_cumsum(x, discount=0.99)
    [4.890798, 4.9402, 3.98, 2.0]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
