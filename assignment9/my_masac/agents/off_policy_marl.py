import os.path
import wandb
import socket
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from torch import Tensor
from torch.distributions import Categorical
from pathlib import Path
from operator import itemgetter
from torch.nn import ModuleDict,Module 
from my_masac.utils.utils import get_time_string, create_directory, space2shape
from my_masac.buffer.buffer import MARL_OffPolicyBuffer
from my_masac.learners.masac_learner import MASAC_Learner

class np2dict_tensor(Module):
    def __init__(self,
                 input_shape,
                 device= None,
                 **kwargs):
        super(np2dict_tensor, self).__init__()
        assert len(input_shape) == 1
        self.output_shapes = {'state': (input_shape[0],)}
        self.device = device

    def forward(self, observations: np.ndarray):
        state = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return {'state': state}
    
class OffPolicyMARLAgents:#(MARLAgents)
    """The core class for off-policy algorithm with multiple agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config,
                 envs):#, SubprocVecMultiAgentEnv
        super(OffPolicyMARLAgents, self).__init__()
        self.action_space = envs.action_space
        self.agent_keys = envs.agents
        self.config = config
        self.device = config.device
        self.distributed_training = config.distributed_training
        self.envs = envs
        # self.envs.reset()
        self.fps = config.fps
        
        self.gamma = config.gamma
        
        self.n_agents = self.config.n_agents = envs.num_agents
        self.n_envs = envs.num_envs
        self.n_epochs = config.n_epochs if hasattr(config, "n_epochs") else 1
        self.observation_space = envs.observation_space
        self.rank = 0
        self.render = config.render
        self.start_training = config.start_training if hasattr(config, "start_training") else 1
        
        self.training_frequency = config.training_frequency if hasattr(config, "training_frequency") else 1
        self.use_rnn = config.use_rnn if hasattr(config, "use_rnn") else False
        self.use_parameter_sharing = config.use_parameter_sharing
        self.use_actions_mask = config.use_actions_mask if hasattr(config, "use_actions_mask") else False
        self.use_global_state = config.use_global_state if hasattr(config, "use_global_state") else False
        
        self.world_size = 1
        # Environment attributes.
        self.state_space = envs.state_space if self.use_global_state else None
        self.episode_length = config.episode_length if hasattr(config, "episode_length") else envs.max_episode_steps
        self.config.episode_length = self.episode_length
        self.current_step = 0
        self.current_episode = np.zeros((self.n_envs,), np.int32)

        
        time_string = get_time_string()
        seed = f"seed_{config.seed}_"
        self.model_dir_load = config.model_dir
        self.model_dir_save = os.path.join(os.getcwd(), config.model_dir, seed + time_string)
        config_dict = vars(config)
        

        # predefine necessary components
        self.model_keys = [self.agent_keys[0]] if self.use_parameter_sharing else self.agent_keys
        self.policy= None
        self.learner= None
        self.memory= None

        self.on_policy = False
        self.start_greedy = config.start_greedy if hasattr(config, "start_greedy") else None
        self.end_greedy = config.end_greedy if hasattr(config, "start_greedy") else None
        self.delta_egreedy= None
        self.e_greedy= None

        self.start_noise = config.start_noise if hasattr(config, "start_noise") else None
        self.end_noise = config.end_noise if hasattr(config, "end_noise") else None
        self.delta_noise= None
        self.noise_scale= None


        self.actions_low = self.action_space.low if hasattr(self.action_space, "low") else None
        self.actions_high = self.action_space.high if hasattr(self.action_space, "high") else None

        self.auxiliary_info_shape = None
        self.memory = None

        self.buffer_size = self.config.buffer_size
        self.batch_size = self.config.batch_size



        log_dir = config.log_dir
        wandb_dir = Path(os.path.join(os.getcwd(), config.log_dir))
        create_directory(str(wandb_dir))
        wandb.init(config=config_dict,
                    project=config.project_name,
                    entity=config.wandb_user_name,
                    notes=socket.gethostname(),
                    dir=wandb_dir,
                    group=config.env_id,
                    job_type=config.agent,
                    name=time_string,
                    reinit=True,
                    settings=wandb.Settings(start_method="fork")
                    )
        # os.environ["WANDB_SILENT"] = "True"
        self.use_wandb = True
        self.log_dir = log_dir

    
                
    def _build_representation(self,input_space) :
        representation = ModuleDict()
        for key in self.model_keys:
            representation[key] = np2dict_tensor(input_shape=space2shape(input_space[key]),device=self.device)
        return representation


    def _build_learner(self, *args):
        return MASAC_Learner(*args)

    def _build_inputs(self,
                      obs_dict,
                      avail_actions_dict = None):
        batch_size = len(obs_dict)
        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
        avail_actions_input = None
        agents_id = None
        obs_input = {k: np.stack([data[k] for data in obs_dict]).reshape(bs, -1) for k in self.agent_keys}
        return obs_input, agents_id, avail_actions_input


    def _build_memory(self):
        """Build replay buffer for models training
        """
        if self.use_actions_mask:
            avail_actions_shape = {key: (self.action_space[key].n,) for key in self.agent_keys}
        else:
            avail_actions_shape = None
        input_buffer = dict(agent_keys=self.agent_keys,
                            state_space=self.state_space if self.use_global_state else None,
                            obs_space=self.observation_space,
                            act_space=self.action_space,
                            n_envs=self.n_envs,
                            buffer_size=self.buffer_size,
                            batch_size=self.batch_size,
                            avail_actions_shape=avail_actions_shape,
                            use_actions_mask=self.use_actions_mask,
                            max_episode_steps=self.episode_length)
        Buffer = MARL_OffPolicyBuffer
        return Buffer(**input_buffer)

    def _build_policy(self) -> Module:
        raise NotImplementedError

    def store_experience(self, obs_dict, avail_actions, actions_dict, obs_next_dict, avail_actions_next,
                         rewards_dict, terminals_dict, info, **kwargs):
        
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_dict]) for k in self.agent_keys},
            'obs_next': {k: np.array([data[k] for data in obs_next_dict]) for k in self.agent_keys},
            'rewards': {k: np.array([data[k] for data in rewards_dict]) for k in self.agent_keys},
            'terminals': {k: np.array([data[k] for data in terminals_dict]) for k in self.agent_keys},
            'agent_mask': {k: np.array([data['agent_mask'][k] for data in info]) for k in self.agent_keys},
        }
        
        if self.use_global_state:
            experience_data['state'] = np.array(kwargs['state'])
            experience_data['state_next'] = np.array(kwargs['next_state'])
        if self.use_actions_mask:
            experience_data['avail_actions'] = {k: np.array([data[k] for data in avail_actions])
                                                for k in self.agent_keys}
            experience_data['avail_actions_next'] = {k: np.array([data[k] for data in avail_actions_next])
                                                     for k in self.agent_keys}
        self.memory.store(**experience_data)

    def exploration(self, batch_size,
                    pi_actions_dict,
                    avail_actions_dict = None):
        if self.e_greedy is not None:
            if np.random.rand() < self.e_greedy:
                if self.use_actions_mask:
                    explore_actions = [{k: Categorical(Tensor(avail_actions_dict[e][k])).sample().numpy()
                                        for k in self.agent_keys} for e in range(batch_size)]
                else:
                    explore_actions = [{k: self.action_space[k].sample() for k in self.agent_keys} for _ in
                                       range(batch_size)]
            else:
                explore_actions = pi_actions_dict
        
        explore_actions = pi_actions_dict
        return explore_actions

    def train(self, n_steps):
        
        return_info = {}
        if self.use_rnn:
            with tqdm(total=n_steps) as process_bar:
                step_start, step_last = deepcopy(self.current_step), deepcopy(self.current_step)
                n_steps_all = n_steps * self.n_envs
                while step_last - step_start < n_steps_all:
                    self.run_episodes(None, n_episodes=self.n_envs, test_mode=False)
                    if self.current_step >= self.start_training:
                        train_info = self.train_epochs(n_epochs=self.n_epochs)
                        self.log_infos(train_info, self.current_step)
                        return_info.update(train_info)
                    process_bar.update((self.current_step - step_last) // self.n_envs)
                    step_last = deepcopy(self.current_step)
                process_bar.update(n_steps - process_bar.last_print_n)
            return return_info

        obs_dict = self.envs.buf_obs
        avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
        state = self.envs.buf_state.copy() if self.use_global_state else None
        for _ in tqdm(range(n_steps)):
            step_info = {}
            policy_out = self.action(obs_dict=obs_dict, avail_actions_dict=avail_actions, test_mode=False)
            actions_dict = policy_out['actions']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = self.envs.step(actions_dict)
            next_state = self.envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = self.envs.buf_avail_actions if self.use_actions_mask else None
            self.store_experience(obs_dict, avail_actions, actions_dict, next_obs_dict, next_avail_actions,
                                  rewards_dict, terminated_dict, info,
                                  **{'state': state, 'next_state': next_state})
            if self.current_step >= self.start_training and self.current_step % self.training_frequency == 0:
                train_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(train_info, self.current_step)
                return_info.update(train_info)
            obs_dict = deepcopy(next_obs_dict)
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
                    if self.use_wandb:
                        step_info[f"Train-Results/Episode-Steps/rank_{self.rank}/env-%d" % i] = info[i]["episode_step"]
                        step_info[f"Train-Results/Episode-Rewards/rank_{self.rank}/env-%d" % i] = info[i]["episode_score"]
                    else:
                        step_info[f"Train-Results/Episode-Steps/rank_{self.rank}"] = {
                            "env-%d" % i: info[i]["episode_step"]}
                        step_info[f"Train-Results/Episode-Rewards/rank_{self.rank}"] = {
                            "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                    self.log_infos(step_info, self.current_step)
                    return_info.update(step_info)

            self.current_step += self.n_envs
        return return_info

    def run_episodes(self, env_fn=None, n_episodes: int = 1, test_mode: bool = False):
       
        envs = self.envs if env_fn is None else env_fn()
        num_envs = envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        episode_count, scores, best_score = 0, [], -np.inf
        obs_dict, info = envs.reset()
        state = envs.buf_state.copy() if self.use_global_state else None
        avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                images = envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)
        else:
            if self.use_rnn:
                self.memory.clear_episodes()
        rnn_hidden = self.init_rnn_hidden(num_envs)

        while episode_count < n_episodes:
            step_info = {}
            policy_out = self.action(obs_dict=obs_dict,
                                     avail_actions_dict=avail_actions,
                                     rnn_hidden=rnn_hidden,
                                     test_mode=test_mode)
            rnn_hidden, actions_dict = policy_out['hidden_state'], policy_out['actions']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = envs.step(actions_dict)
            next_state = envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
            if test_mode:
                if self.config.render_mode == "rgb_array" and self.render:
                    images = envs.render(self.config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
            else:
                self.store_experience(obs_dict, avail_actions, actions_dict, next_obs_dict, next_avail_actions,
                                      rewards_dict, terminated_dict, info,
                                      **{'state': state, 'next_state': next_state})
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
                    
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    if test_mode:
                        if best_score < episode_score:
                            best_score = episode_score
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (episode_count, episode_score))
                    else:
                        if self.use_wandb:
                            step_info["Train-Results/Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                            step_info["Train-Results/Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                        else:
                            step_info["Train-Results/Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                            step_info["Train-Results/Episode-Rewards"] = {
                                "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                        self.current_step += info[i]["episode_step"]
                        self.log_infos(step_info, self.current_step)

        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                # time, height, width, channel -> time, channel, height, width
                videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

            if self.config.test_mode:
                print("Best Score: %.2f" % best_score)

            test_info = {
                "Test-Results/Episode-Rewards": np.mean(scores),
                "Test-Results/Episode-Rewards-Std": np.std(scores),
            }

            self.log_infos(test_info, self.current_step)
            if env_fn is not None:
                envs.close()
        return scores

    def train_epochs(self, n_epochs=1):
        
        info_train = {}
        for i_epoch in range(n_epochs):
            sample = self.memory.sample()
            info_train = self.learner.update(sample)
        info_train["epsilon-greedy"] = self.e_greedy
        info_train["noise_scale"] = self.noise_scale
        return info_train

    def test(self, env_fn, n_episodes):
        
        scores = self.run_episodes(env_fn=env_fn, n_episodes=n_episodes, test_mode=True)
        return scores
    

    def save_model(self, model_name):
        # save the neural networks
        if not os.path.exists(self.model_dir_save):
            os.makedirs(self.model_dir_save)
        model_path = os.path.join(self.model_dir_save, model_name)
        self.learner.save_model(model_path)

    def load_model(self, path, model=None):
        # load neural networks
        self.learner.load_model(path, model)

    def log_infos(self, info: dict, x_index: int):
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        for k, v in info.items():
            if v is None:
                continue
            wandb.log({k: v}, step=x_index)

    def log_videos(self, info: dict, fps: int, x_index: int = 0):
        for k, v in info.items():
            if v is None:
                continue
            wandb.log({k: wandb.Video(v, fps=fps, format='gif')}, step=x_index)
    

    def finish(self):
        wandb.finish()