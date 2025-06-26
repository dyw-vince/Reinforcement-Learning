import importlib
import numpy as np
# from my_masac.environment import RawMultiAgentEnv
from pettingzoo.mpe import simple_adversary_v3, simple_tag_v3
TEAM_NAME_DICT = {
    "mpe.simple_adversary_v3": ['adversary', 'agent']
}


class MPE_Env:  # (RawMultiAgentEnv)
    """
    The implementation of MPE environments, provides a standardized interface for interacting
    with the environments in the context of multi-agent reinforcement learning.

    Parameters:
        config: The configurations of the environment.
    """

    def __init__(self, config):
        super(MPE_Env, self).__init__()
        # Prepare raw environment
        # env_name, env_id = config.env_name, config.env_id
        self.config = config
        self.render_mode = config.render_mode
        self.continuous_actions = config.continuous_action
        # self.scenario_name = env_name + "." + env_id
        # scenario = importlib.import_module(f'pettingzoo.{env_name}.{env_id}')  # create scenario
        self.env = simple_tag_v3.parallel_env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            continuous_actions=self.continuous_actions,
            render_mode=self.render_mode
        )
        self.env.reset(config.env_seed)

        # Set basic attributes
        self.agents = self.env.agents
        self.metadata = self.env.metadata
        self.num_agents = self.env.num_agents
        self.state_space = self.env.state_space
        self.observation_space = {
            agent: self.env.observation_space(agent) for agent in self.agents}
        self.action_space = {agent: self.env.action_space(
            agent) for agent in self.agents}
        self.max_episode_steps = self.env.unwrapped.max_cycles
        self.agent_groups = [
            [agent for agent in self.agents if "adversary" in agent],
            [agent for agent in self.agents if "agent" in agent]
        ]
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

        # reset_info["episode_step"] = self._episode_step  # current episode step
        # reset_info["episode_score"] = self.individual_episode_reward  # the accumulated rewards
        # reset_info["agent_mask"] = self.agent_mask()
        # reset_info["avail_actions"] = self.avail_actions()
        # reset_info["state"] = self.state()

        return observations, reset_info

    def step(self, actions):
        """Take an action as input, perform a step in the underlying pettingzoo environment."""
        if self.continuous_actions:
            for k, v in actions.items():
                actions[k] = np.clip(
                    v, self.action_space[k].low, self.action_space[k].high)
        observations, rewards, terminated, truncated, info = self.env.step(
            actions)

        # 添加辅助奖励
        if not self.config.test_mode:
            agent_positions = self.env.state()
            pos_dict = {agent: agent_positions[2+17*i:4+17*i]
                        for i, agent in enumerate(self.agents)}
            # print(pos_dict)

            for adv in self.agent_groups[0]:
                adv_pos = np.array(pos_dict[adv])
                # 计算到所有 prey 的平均距离
                mean_dist = np.mean(
                    [np.linalg.norm(adv_pos - np.array(pos_dict[prey])) for prey in self.agent_groups[1]])
                shaped_reward = -0.01 * mean_dist  # 可以调节系数
                rewards[adv] += shaped_reward
                info[adv]['shaped_reward'] = shaped_reward  # 可视化时记录

        for k, v in rewards.items():
            self.individual_episode_reward[k] += v
        step_info = {"infos": info,
                     "individual_episode_rewards": self.individual_episode_reward}
        self._episode_step += 1
        truncated = True if self._episode_step >= self.max_episode_steps else False

        # step_info["episode_step"] = self._episode_step  # current episode step
        # step_info["episode_score"] = self.individual_episode_reward  # the accumulated rewards
        # step_info["agent_mask"] = self.agent_mask()
        # step_info["avail_actions"] = self.avail_actions()
        # step_info["state"] = self.state()
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
