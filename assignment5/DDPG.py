import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from replay_buffer import ReplayBuffer  # 你原来的版本可以继续用
from utils import preprocess,NoiseGenerator  # 保留你已有的预处理方式
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(Actor, self).__init__()
        h,w,c = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=3),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, shape[2], shape[0], shape[1]))  # (B, C, H, W)
        # o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x=x.permute(0,3,1,2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x=self.fc(x)
        return x

class Critic(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(Critic, self).__init__()
        h, w,c = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=3),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + action_dim , 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, shape[2], shape[0], shape[1]))  # (B, C, H, W)
        # o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, action):
        x=x.permute(0,3,1,2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        return self.fc(x)

class AgentDDPG:
    def __init__(self, action_space, model_outputs=None, noise_mean=None, noise_std=None):
        self.gamma = 0.99
        self.tau = 0.005
        self.actor_lr = 1e-5
        self.critic_lr = 2e-3
        self.memory_capacity = 100000

        self.need_decode_out = model_outputs is not None
        self.model_action_out = model_outputs if model_outputs else action_space.shape[0]
        self.action_space = action_space

        self.noise = NoiseGenerator(
            noise_mean if noise_mean is not None else np.zeros(self.model_action_out),
            noise_std if noise_std is not None else np.ones(self.model_action_out) * 0.2
        )

        self.buffer = ReplayBuffer(self.memory_capacity)
        self.actor = None
        self.critic = None
        self.target_actor = None
        self.target_critic = None

    def reset(self):
        self.noise.reset()

    def init_networks(self, input_shape):
        self.actor = Actor(input_shape, self.model_action_out).to(device)
        self.critic = Critic(input_shape, self.model_action_out).to(device)

        self.target_actor = Actor(input_shape, self.model_action_out).to(device)
        self.target_critic = Critic(input_shape, self.model_action_out).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def get_action(self, state, add_noise=True):
        prep_state = preprocess(state)
        if self.actor is None:
            self.init_networks(prep_state.shape)

        state_tensor = torch.FloatTensor(prep_state).unsqueeze(0).to(device)
        action = self.actor(state_tensor).detach().cpu().numpy()[0]
        
        if add_noise:
            action += self.noise.generate()
        if self.need_decode_out:
            env_action = self.decode_model_output(action)
        else:
            env_action = action
        return np.clip(env_action, self.action_space.low, self.action_space.high), action

    def decode_model_output(self, model_out):
        return np.array([model_out[0], np.clip(model_out[1], 0, 1), -np.clip(model_out[1], -1, 0)])

    def learn(self, state, train_action, reward, new_state):
        s = preprocess(state)
        s_ = preprocess(new_state)
        self.buffer.save_move(s, train_action, reward, s_)
        if self.buffer.transitions_num < 64:
            return

        state_batch, action_batch, reward_batch, new_state_batch = self.buffer.sample_buffer()
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        new_state_batch = torch.FloatTensor(new_state_batch).to(device)
        # Update critic
        with torch.no_grad():
            next_actions = self.target_actor(new_state_batch)
            target_q = self.target_critic(new_state_batch, next_actions)
            target = reward_batch + self.gamma * target_q
        critic_loss = F.mse_loss(self.critic(state_batch, action_batch), target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_network(self.target_actor, self.actor)
        self.update_target_network(self.target_critic, self.critic)

    def update_target_network(self, target_net, source_net):
        for t, s in zip(target_net.parameters(), source_net.parameters()):
            t.data.copy_(self.tau * s.data + (1.0 - self.tau) * t.data)

    def save_model(self, path='models/', suffix=''):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, f'{suffix}actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, f'{suffix}critic.pth'))
        torch.save(self.target_actor.state_dict(), os.path.join(path, f'{suffix}target_actor.pth'))
        torch.save(self.target_critic.state_dict(), os.path.join(path, f'{suffix}target_critic.pth'))

    def load_model(self, path='models/', suffix=''):
        if self.actor is None or self.critic is None:
            # 需要用一个 state 形状来初始化网络
            dummy_state = np.zeros((96, 96, 3))  # 例如 CarRacing 的图像输入
            self.init_networks(dummy_state.shape)
        self.actor.load_state_dict(torch.load(os.path.join(path, f'{suffix}actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(path, f'{suffix}critic.pth')))
        self.target_actor.load_state_dict(torch.load(os.path.join(path, f'{suffix}target_actor.pth')))
        self.target_critic.load_state_dict(torch.load(os.path.join(path, f'{suffix}target_critic.pth')))
