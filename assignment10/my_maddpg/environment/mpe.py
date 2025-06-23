import importlib
import numpy as np
from gym import spaces
from pettingzoo.mpe import simple_adversary_v3

class MPE_Env:
    

    def __init__(self, config):
        super(MPE_Env, self).__init__()
        self.render_mode = config.render_mode
        self.continuous_actions = config.continuous_action
        self.env = simple_adversary_v3.parallel_env(continuous_actions=self.continuous_actions, render_mode=self.render_mode)
        self.env.reset(config.env_seed)
        
        # Set basic attributes
        self.agents = self.env.agents
        self.metadata = self.env.metadata
        self.num_agents = self.env.num_agents
        self.state_space = self.env.state_space
        self.observation_space = {agent: self.env.observation_space(agent) for agent in self.agents}
        self.action_space = {agent: self.env.action_space(agent) for agent in self.agents}
        self.max_episode_steps = self.env.unwrapped.max_cycles
        self.agent_groups = [['adversary_0'], ['agent_0', 'agent_1']]
        self.individual_episode_reward = {k: 0.0 for k in self.agents}
        self.episode_score = {k: 0.0 for k in self.agents}
        self._episode_step = 0
        self.env_info = self.get_env_info()
        self.groups_info = self.get_groups_info()
    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, *args):
        """Get the rendered images of the environment."""
        return self.env.render()

    def reset(self):
        """Reset the environment to its initial state."""
        observations, infos = self.env.reset()
        for agent_key in self.agents:
            self.individual_episode_reward[agent_key] = 0.0
        reset_info = {"infos": infos,
                      "individual_episode_rewards": self.individual_episode_reward}
        self._episode_step = 0


        return observations, reset_info

    def step(self, actions):
        """Take an action as input, perform a step in the underlying pettingzoo environment."""
        if self.continuous_actions:
            for k, v in actions.items():
                actions[k] = np.clip(v, self.action_space[k].low, self.action_space[k].high)
        observations, rewards, terminated, truncated, info = self.env.step(actions)
        for k, v in rewards.items():
            self.individual_episode_reward[k] += v
        step_info = {"infos": info,
                     "individual_episode_rewards": self.individual_episode_reward}
        self._episode_step += 1
        truncated = True if self._episode_step >= self.max_episode_steps else False

        return observations, rewards, terminated, truncated, step_info

    def state(self):
        """Returns the global state of the environment."""
        return self.env.state()

    def agent_mask(self):
        """
        Create a boolean mask indicating which agents are currently alive.
        Note: For MPE environment, all agents are alive before the episode is terminated.
        """
        return {agent: True for agent in self.agents}

    def avail_actions(self):
        """Returns a boolean mask indicating which actions are available for each agent."""
        if self.continuous_actions:
            return None
        else:
            return {agent: np.ones(self.action_space[agent].n, np.bool_) for agent in self.agents}

    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.num_agents,
                'max_episode_steps': self.max_episode_steps}

    def get_groups_info(self):
        return {'num_groups': len(self.agent_groups),
                'agent_groups': self.agent_groups,
                'observation_space_groups': [{k: self.observation_space[k] for i, k in enumerate(group)}
                                             for group in self.agent_groups],
                'action_space_groups': [{k: self.action_space[k] for i, k in enumerate(group)}
                                        for group in self.agent_groups],
                'num_agents_groups': [len(group) for group in self.agent_groups]}
    


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
    