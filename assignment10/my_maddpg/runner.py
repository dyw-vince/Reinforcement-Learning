
import yaml
import numpy as np
from copy import deepcopy
from types import SimpleNamespace as SN
import os
from argparse import Namespace
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from operator import itemgetter
from my_maddpg.environment import make_envs
from my_maddpg.utils.utils import combine_actions
from my_maddpg.utils.utils import set_seed
from my_maddpg.agents.maddpg_agents import MADDPG_Agents


EPS = 1e-8


def get_arguments(method, env, env_id, config_path=None, parser_args=None, is_test=False):
    """Get arguments from .yaml files
    Args:
        method: the algorithm name that will be implemented,
        env: The name of the environment,
        env_id: The name of the scenario in the environment.
        config_path: default is None, if None, the default configs (xuance/configs/.../*.yaml) will be loaded.
        parser_args: arguments that specified by parser tools.

    Returns:
        args: the SimpleNamespace variables that contains attributes for DRL implementations.
    """
    config_path = "my_maddpg/configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    configs = []
    for i, agent in enumerate(method):
        configs.append(config)
    # 将字典比如arg['device']变成arg.device
    args = [SN(**config_i) for config_i in configs]

    if is_test:  # for test mode
        for i_args in range(len(args)):
            args[i_args].test_mode = int(is_test)
            args[i_args].parallels = 1
    return args


def get_runner(method, env, env_id, config_path=None, parser_args=None, is_test=False):

    args = get_arguments(method, env, env_id, config_path,
                         parser_args, is_test)
    device = args[0].device
    print(f"Calculating device: {device}")
    if type(args) == list:
        agents_name_string = []
        for i in range(len(method)):
            if i < len(method) - 1:
                agents_name_string.append(args[i].agent + " vs")
            else:
                agents_name_string.append(args[i].agent)
            args[i].agent_name = method[i]
            args[i].model_dir = f"models/model_{i}"
            args[i].log_dir = f"logs/log_{i}"
        print("Algorithm:", *agents_name_string)
        print("Environment:", args[0].env_name)
        print("Scenario:", args[0].env_id)
        runner_name = args[0].runner
        runner = Runner(args)
        return runner


class Runner(object):
    def __init__(self, configs):
        self.configs = configs
        set_seed(self.configs[0].seed)

        # build environments
        self.envs = make_envs(self.configs[0])  # self.configs[0]
        self.n_envs = self.envs.num_envs
        self.current_step = 0
        self.envs.reset()
        self.groups_info = self.envs.groups_info
        self.groups = self.groups_info['agent_groups']
        self.num_groups = self.groups_info['num_groups']
        self.obs_space_groups = self.groups_info['observation_space_groups']
        self.act_space_groups = self.groups_info['action_space_groups']
        assert len(
            configs) == self.num_groups, "Number of groups must be equal to the number of methods."
        self.agents = []
        for group in range(self.num_groups):
            _env = Namespace(num_agents=len(self.groups[group]),
                             num_envs=self.envs.num_envs,
                             agents=self.groups[group],
                             state_space=self.envs.state_space,
                             observation_space=self.obs_space_groups[group],
                             action_space=self.act_space_groups[group],
                             max_episode_steps=self.envs.max_episode_steps)
            self.agents.append(MADDPG_Agents(self.configs[group], _env))

        self.distributed_training = self.agents[0].distributed_training
        self.use_actions_mask = self.agents[0].use_actions_mask
        self.use_global_state = self.agents[0].use_global_state
        self.use_rnn = self.agents[0].use_rnn
        self.use_wandb = self.agents[0].use_wandb

        self.rank = 0

    def rprint(self, info: str):
        if self.rank == 0:
            print(info)

    def run(self):
        if self.configs[0].test_mode:
            def env_fn():
                config_test = deepcopy(self.configs[0])
                config_test.parallels = 1
                config_test.render = True
                return make_envs(config_test)

            for agent in self.agents:
                agent.render = True
                agent.load_model(agent.model_dir_load)

            scores = self.test(
                env_fn, self.configs[0].test_episode, test_mode=True)

            print(f"Mean Score: {scores}, Std: {scores}")
            print("Finish testing.")
        else:
            n_train_steps = self.configs[0].running_steps // self.n_envs

            self.train(n_train_steps)

            print("Finish training.")
            for agent in self.agents:
                agent.save_model("final_train_model.pth")

        for agent in self.agents:
            agent.finish()
        self.envs.close()

    def train(self, n_steps):
        """
        Train the model for numerous steps.
        Args:
            n_steps (int): Number of steps to train the model:
        """

        obs_dict = self.envs.buf_obs
        avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
        state = self.envs.buf_state.copy() if self.use_global_state else None
        for _ in tqdm(range(n_steps)):
            step_info = {}
            policy_out_list = [agent.action(obs_dict=obs_dict,
                                            state=state,
                                            avail_actions_dict=avail_actions,
                                            test_mode=False) for agent in self.agents]
            actions_execute = combine_actions(policy_out_list, self.n_envs)
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(
                actions_execute)
            next_state = self.envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = self.envs.buf_avail_actions.copy(
            ) if self.use_actions_mask else None
            for agent in self.agents:
                agent.store_experience(obs_dict, avail_actions, actions_execute, next_obs_dict, next_avail_actions,
                                       rewards_dict, terminated_dict, info,
                                       **{'state': state, 'next_state': next_state})
                if self.current_step >= agent.start_training and self.current_step % agent.training_frequency == 0:
                    train_info = agent.train_epochs(n_epochs=agent.n_epochs)
                    agent.log_infos(train_info, self.current_step)

            obs_dict = next_obs_dict

            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(self.n_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    obs_dict[i] = info[i]["reset_obs"]
                    self.envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        self.envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        self.envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    for agent in self.agents:
                        episode_score = np.mean(itemgetter(
                            *agent.agent_keys)(info[i]["episode_score"]))
                        if self.use_wandb:
                            step_info[f"Train-Results/Episode-Steps/rank_{self.rank}/env-%d" % i] = info[i][
                                "episode_step"]
                            step_info[f"Train-Results/Episode-Rewards/rank_{self.rank}/env-%d" %
                                      i] = episode_score
                        else:
                            step_info[f"Train-Results/Episode-Steps/rank_{self.rank}"] = {
                                "env-%d" % i: info[i]["episode_step"]}
                            step_info[f"Train-Results/Episode-Rewards/rank_{self.rank}"] = {
                                "env-%d" % i: episode_score}
                        agent.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            for agent in self.agents:
                agent.current_step += self.n_envs

    def test(self, env_fn=None, n_episodes: int = 1, test_mode: bool = False):
        """
        Run some episodes when use RNN.

        Parameters:
            env_fn: The function that can make some testing environments.
            n_episodes (int): Number of episodes.
            test_mode (bool): Whether to test the model.

        Returns:
            Scores: The episode scores.
        """
        envs = self.envs if env_fn is None else env_fn()
        num_envs = envs.num_envs
        videos = [[] for _ in range(num_envs)]
        episode_videos = [[] for _ in range(self.num_groups)]
        episode_count = 0
        scores = [[0.0 for _ in range(num_envs)]
                  for _ in range(self.num_groups)]
        best_score = [-np.inf for _ in range(self.num_groups)]
        obs_dict, info = envs.reset()
        state = envs.buf_state.copy() if self.use_global_state else None
        avail_actions = envs.buf_avail_actions if self.use_actions_mask else None

        for config in self.configs:
            if config.render_mode == "rgb_array" and config.render:
                images = envs.render(config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)
        # rnn_hidden = [agent.init_rnn_hidden(num_envs) for agent in self.agents]

        while episode_count < n_episodes:
            step_info = {}
            policy_out_list = [agent.action(obs_dict=obs_dict,
                                            state=state,
                                            avail_actions_dict=avail_actions,
                                            # rnn_hidden=rnn_hidden[i_agt],
                                            test_mode=test_mode) for i_agt, agent in enumerate(self.agents)]
            actions_execute = combine_actions(policy_out_list, num_envs)
            # rnn_hidden = [policy_out['hidden_state'] for policy_out in policy_out_list]
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = envs.step(
                actions_execute)
            next_state = envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = envs.buf_avail_actions if self.use_actions_mask else None

            for config in self.configs:
                if config.render_mode == "rgb_array" and config.render:
                    images = envs.render(config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
            obs_dict = deepcopy(next_obs_dict)
            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(num_envs):

                if all(terminated_dict[i].values()) or truncated[i]:
                    episode_count += 1
                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        self.envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]

                    for i_group in range(self.num_groups):
                        episode_score = float(
                            np.mean(itemgetter(*self.groups[i_group])(info[i]["episode_score"])))
                        scores[i_group].append(episode_score)
                        if test_mode:
                            if best_score[i_group] < episode_score:
                                best_score[i_group] = episode_score
                                episode_videos[i_group] = videos[i].copy()
                            if self.configs[i_group].test_mode:
                                print("Episode: %d, Score: %.2f" %
                                      (episode_count, episode_score))
                        else:
                            for agent in self.agents:
                                episode_score = np.mean(itemgetter(
                                    *agent.agent_keys)(info[i]["episode_score"]))
                                if self.use_wandb:
                                    step_info["Train-Results/Episode-Steps/env-%d" %
                                              i] = info[i]["episode_step"]
                                    step_info["Train-Results/Episode-Rewards/env-%d" %
                                              i] = episode_score
                                else:
                                    step_info["Train-Results/Episode-Steps"] = {
                                        "env-%d" % i: info[i]["episode_step"]}
                                    step_info["Train-Results/Episode-Rewards"] = {
                                        "env-%d" % i: episode_score}
                                agent.log_infos(step_info, self.current_step)
                                if not agent.on_policy:
                                    agent._update_explore_factor()

        for i_group in range(self.num_groups):
            config = self.configs[i_group]
            if config.render_mode == "rgb_array" and config.render:
                # time, height, width, channel -> time, channel, height, width
                videos_info = {"Videos_Test": np.array([episode_videos[i_group]],
                                                       dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                self.agents[i_group].log_videos(
                    info=videos_info, fps=config.fps, x_index=self.current_step)

        if self.configs[0].test_mode:
            print("Best Score: ", best_score)

        for i_group in range(self.num_groups):
            test_info = {
                "Test-Results/Episode-Rewards": np.mean(scores[i_group]),
                "Test-Results/Episode-Rewards-Std": np.std(scores[i_group]),
            }

            self.agents[i_group].log_infos(test_info, self.current_step)
        if env_fn is not None:
            envs.close()
        return scores
