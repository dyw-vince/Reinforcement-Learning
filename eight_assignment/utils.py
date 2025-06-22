import numpy as np
import torch.nn.functional as F
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

class QNet_RNN(nn.Module):
    def __init__(self, config, Qnet_input_dim):
        super().__init__()
        self.hidden_dim=config['rnn_hidden_dim']
        self.device=config['device']
        self.rnn_hidden=None
        self.fc1 = nn.Linear(Qnet_input_dim, config['rnn_hidden_dim'])
        self.rnn = nn.LSTM(config['rnn_hidden_dim'],  config['rnn_hidden_dim'],batch_first=True)
        self.fc2 = nn.Linear( config['rnn_hidden_dim'], config['action_dim'])
        if config['use_orthogonal_init']:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)
    def init_hidden(self,batch_size):
        h = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)
        self.rnn_hidden=(h, c)

    def forward(self, inputs):
            # When 'choose_action', inputs.shape(N,input_dim)
            # When 'train', inputs.shape(bach_size*N,input_dim)
            x = F.relu(self.fc1(inputs))
            x = x.unsqueeze(1)        
            output,self.rnn_hidden =self.rnn(x, self.rnn_hidden)
            Q = self.fc2(output[:,-1,:])
            return Q
class RNN(nn.Module):
    """A simple RNN for the agent's Q-network."""
    def __init__(self,config, Qnet_input_dim):
        super(RNN, self).__init__()
        self.input_dim=Qnet_input_dim
        self.hidden_dim=config['rnn_hidden_dim']
        self.output_dim=config['action_dim']
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3,gain=0.01)
    def forward(self, inputs):
        """Forward pass through the network."""
        x = self.norm1(F.tanh(self.fc1(inputs)))
        x = self.norm2(F.tanh(self.fc2(x)))
        x = F.tanh(self.fc(x))
        return self.fc3(x)    
# class QNet_MLP(nn.Module):
#     def __init__(self, config, Qnet_input_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(Qnet_input_dim, config['rnn_hidden_dim'])
#         self.norm1=nn.LayerNorm(config['hidden_dim'])
#         self.fc2=nn.Linear(config['hidden_dim'],config['hidden_dim'])
#         self.norm2=nn.LayerNorm(config['hidden_dim'])
#         self.fc3 = nn.Linear(config['hidden_dim'], config['action_dim'])
#         #[32, 64, 64, 128],
#     def forward(self, inputs):
#         x=self.norm1(F.relu(self.fc1(inputs)))
#         x=self.norm2(F.relu(self.fc2(x)))
#         return self.fc3(x)
class Qmix(nn.Module):
    """The Q-mixing network."""
    def __init__(self,config):
        super(Qmix, self).__init__()
        self.state_dim=config['global_state_dim']
        self.hidden_dim=config['hidden_dim']
        self.N=config['N']
        self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.N))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1))
        # Use Softplus to ensure weights are non-negative
        self.trans_fn = nn.Softplus(beta=1, threshold=20)
        orthogonal_init(self.hyper_w1)
        orthogonal_init(self.hyper_b1)
        
    def forward(self, qs, states):
        """Forward pass for the mixing network."""
        weight = self.trans_fn(self.hyper_w1(states))
        bias = self.hyper_b1(states)
        return torch.sum(weight * qs, dim=-1, keepdim=True) + bias
# class Q_MIX_Net(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.N=config['N']
#         self.global_state_dim=config['global_state_dim']
#         self.Qmix_hidden_dim=config['Qmix_hidden_dim']
#         self.hyper_hidden_dim=config['hyper_hidden_dim']
#         self.batch_size=config['batch_size']
#         self.hyper_layers_num=config['hyper_layers_num']
#         if self.hyper_layers_num == 2:
#             self.hyper_w1=nn.Sequential(
#                 nn.Linear(self.global_state_dim,self.hyper_hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(self.hyper_hidden_dim,self.N*self.Qmix_hidden_dim)
#             )
#             self.hyper_w2=nn.Sequential(
#                 nn.Linear(self.global_state_dim,self.hyper_hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(self.hyper_hidden_dim,self.Qmix_hidden_dim)
#             )
#         elif self.hyper_layers_num == 1:
#             print("hyper_layers_num=1")
#             self.hyper_w1 = nn.Linear(self.global_state_dim, self.N * self.Qmix_hidden_dim)
#             self.hyper_w2 = nn.Linear(self.global_state_dim, self.Qmix_hidden_dim)
#         self.b1=nn.Linear(self.global_state_dim,self.Qmix_hidden_dim)
#         self.b2=nn.Sequential(
#             nn.Linear(self.global_state_dim,self.Qmix_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.Qmix_hidden_dim,1)
#         )
#         self.trans_fn=nn.Softplus(beta=1, threshold=20)
#     def forward(self,Qs,global_states):
#         """
#         Qs=[B,episode_max,N]  global_states=[B,episode_max,global_state_dim]
#         -> [B*episode_max,1,N]                [B*episode_max,global_state_dim]
#                 x1=[B*episode_max,1,N]    w1=[B*episode_max,N,Qmix_hidden_dim] b1=[B*episode_max,1,Qmix_hidden_dim]
#                 x2=w1*x1+b1=[B*episode_max,1,Qmix_hidden_dim]
#                 w2=[B,Qmix_hidden_dim,1] b2=[B,1,1]
#                 y=w2*x2+b2=[B,1,1]
#         """
#         Qs=Qs.view(-1, 1,self.N)
#         global_states=global_states.reshape(-1,self.global_state_dim)
#         w1=torch.abs(self.hyper_w1(global_states)).view(-1,self.N,self.Qmix_hidden_dim) 
#         b1=self.b1(global_states).view(-1,1,self.Qmix_hidden_dim)
#         w2=torch.abs(self.hyper_w2(global_states)).view(-1,self.Qmix_hidden_dim,1)
#         b2=self.b2(global_states).view(-1,1,1)

#         Qs_hidden=F.elu(torch.bmm(Qs,w1)+b1)
#         y2=(torch.bmm(Qs_hidden,w2)+b2).reshape(self.batch_size,-1,1)
#         return y2
    
class Buffer:
    #dict(tensor))
    def __init__(self,config):
        
        self.action_dim=config['action_dim']
        self.batch_size = config['batch_size']
        self.buffer = None
        self.buffer_size=config['buffer_size']
        self.current_size=0
        self.device=config['device']
        self.episode_num = 0
        self.episode_len = np.zeros(self.buffer_size)
        self.global_state_dim = config['global_state_dim']
        self.max_episode_times = config['max_episode_times']
        self.N = config['N']
        self.state_dim = config['state_dim']
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'states_n': np.empty([self.buffer_size, self.max_episode_times+1, self.N, self.state_dim]),
                       'global_states': np.empty([self.buffer_size, self.max_episode_times+1, self.global_state_dim]),
                       'actions_n': np.empty([self.buffer_size, self.max_episode_times, self.N]),
                       'onehot_actions_n':np.empty([self.buffer_size, self.max_episode_times+1, self.N , self.action_dim]),
                       'rewards_n': np.empty([self.buffer_size, self.max_episode_times, self.N]),
                       'dones_n': np.empty([self.buffer_size, self.max_episode_times, self.N])
                       }
        self.episode_num = 0

    def store(self,episode_step, states_n, global_state,onehot_actions_n, actions_n, rewards_n, dones_n):
        self.buffer['states_n'][self.episode_num][episode_step] = states_n
        self.buffer['global_states'][self.episode_num][episode_step] = global_state
        self.buffer['onehot_actions_n'][self.episode_num][episode_step+1] = onehot_actions_n
        #第一次为全0，第一次得到的结果放在第二位
        self.buffer['actions_n'][self.episode_num][episode_step] = actions_n
        self.buffer['rewards_n'][self.episode_num][episode_step] = rewards_n
        self.buffer['dones_n'][self.episode_num][episode_step] = dones_n

    def store_last_state(self, episode_step, states_n, global_state):
        #使用后向视角计算时，value需要多一位
        self.buffer['states_n'][self.episode_num][episode_step] = states_n
        self.buffer['global_states'][self.episode_num][episode_step] = global_state
        self.episode_len[self.episode_num] = episode_step
        self.episode_num = (self.episode_num + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def get_training_data(self):
        # Randomly sampling
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch = {}
        for key in self.buffer.keys():
            if key == 'states_n' or key == 'global_states' or key == 'onehot_actions_n':
                batch[key] = torch.tensor(self.buffer[key][index, :self.max_episode_times + 1], dtype=torch.float32).to(self.device)
            elif key == 'actions_n':
                batch[key] = torch.tensor(self.buffer[key][index, :self.max_episode_times], dtype=torch.long).to(self.device)
                #action一般作为索引必须为torch.long类型
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :self.max_episode_times], dtype=torch.float32).to(self.device)
        return batch
        # , max_episode_len
    
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

def get_inputs(batch,config):
    inputs = []
    inputs.append(batch['states_n']) #(batch_size,max_episode_len+1,N,states_dim)
    if config['add_last_action']:
        inputs.append(batch['onehot_actions_n']) #(batch_size,max_episode_len+1,N,actions_dim)
    if config['add_agent_id']:
        agent_id_one_hot = torch.eye(config['N']).unsqueeze(0).unsqueeze(0).repeat(config['batch_size'], config['max_episode_times'] + 1, 1, 1).to(config['device'])
        inputs.append(agent_id_one_hot) #(batch_size,max_episode_len+1,N,N)

    # inputs.shape=(batch_size,max_episode_len+1,N,input_dim)
    inputs = torch.cat([x for x in inputs], dim=-1).to(config['device'])
    return inputs

def dict2np(diction):
    return np.array([diction[agent] for agent in diction.keys()])