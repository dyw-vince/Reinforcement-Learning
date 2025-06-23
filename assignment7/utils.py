import numpy as np
from pettingzoo.mpe import simple_spread_v3
import torch
import torch.nn as nn


def make_env(episode_limit, render_mode="None"):
    env = simple_spread_v3.parallel_env(N=3, max_cycles=episode_limit,
            local_ratio=0.5, render_mode=render_mode, continuous_actions=False)
    env.reset(seed=42)
    return env

def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)
            
class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class Actor(nn.Module):
    def __init__(self,config,actor_input_dim):
        super().__init__()
        self.fc1=nn.Linear(actor_input_dim, config['hidden_dim'])
        self.fc2=nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.fc3=nn.Linear(config['hidden_dim'], config['action_dim'])
        self.activate=nn.Tanh()
        if config['use_orthogonal_init']:
            print("------actor_use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)
    def forward(self, x):
        x=self.activate(self.fc1(x))
        x=self.activate(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob

class Critic(nn.Module):
    def __init__(self,config,critic_input_dim):
        super().__init__()
        self.fc1=nn.Linear(critic_input_dim, config['hidden_dim'])
        self.fc2=nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.fc3=nn.Linear(config['hidden_dim'],1)
        self.activate=nn.Tanh()
        if config['use_orthogonal_init']:
            print("------critic_use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)
    def forward(self, x):
        x=self.activate(self.fc1(x))
        x=self.activate(self.fc2(x))
        return self.fc3(x).squeeze(-1)

class Buffer:
    #dict(tensor))
    def __init__(self,config):
        self.N = config['N']
        self.state_dim = config['state_dim']
        self.global_state_dim = config['global_state_dim']
        self.episode_times = config['episode_times']
        self.batch_size = config['batch_size']
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()
    def reset_buffer(self):
        self.buffer = {'states_n': np.empty([self.batch_size, self.episode_times, self.N, self.state_dim]),
                       'global_states': np.empty([self.batch_size, self.episode_times, self.global_state_dim]),
                       'values_n': np.empty([self.batch_size, self.episode_times + 1, self.N]),
                       'actions_n': np.empty([self.batch_size, self.episode_times, self.N]),
                       'logprobs_n': np.empty([self.batch_size, self.episode_times, self.N]),
                       'rewards_n': np.empty([self.batch_size, self.episode_times, self.N]),
                       'dones_n': np.empty([self.batch_size, self.episode_times, self.N])
                       }
        self.episode_num = 0

    def store(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n):
        self.buffer['states_n'][self.episode_num][episode_step] = obs_n
        self.buffer['global_states'][self.episode_num][episode_step] = s
        self.buffer['values_n'][self.episode_num][episode_step] = v_n
        self.buffer['actions_n'][self.episode_num][episode_step] = a_n
        self.buffer['logprobs_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['rewards_n'][self.episode_num][episode_step] = r_n
        self.buffer['dones_n'][self.episode_num][episode_step] = done_n
    def store_last_value(self, episode_step, v_n):
        #使用后向视角计算时，value需要多一位
        self.buffer['values_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1
    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            # if key == 'a_n':
            #     batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            # else:
            batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch
    
def advantage_and_V(batch,config):#后向视角计算,TD0容易高方差更新
    #batch['reward_n']:[B,episode_times,N]
    #batch['value_n']:[B,episode_times,N]
    advantages=[]
    advantage=0
    with torch.no_grad():
        deltas = batch['rewards_n'] + config['gamma']* batch['values_n'][:, 1:] * (1 - batch['dones_n']) - batch['values_n'][:, :-1]
        for t in reversed(range(config['max_episode_times'])):
            advantage = deltas[:, t] + config['gamma'] * config['lambda'] * advantage
            advantages.insert(0, advantage)
        advantages = torch.stack(advantages, dim=1)  # adv.shape(batch_size,episode_limit,N)
        vs = advantages + batch['values_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
        if config['use_adv_norm']:  # Trick 1: advantage normalization
            advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-5))
    return advantages,vs

def get_inputs(batch,N,device):
    #batch['states_n']:[B,episode_times,N,state_dim]
    #batch['global_states']:[B,episode_times,global_state_dim]->[B,episode_times,N,global_state_dim]
    actor_inputs=batch['states_n'].to(device)
    critic_inputs=batch['global_states'].unsqueeze(2).repeat(1, 1, N, 1).to(device)
    return actor_inputs, critic_inputs

def dict2np(diction):
    return np.array([diction[agent] for agent in diction.keys()])