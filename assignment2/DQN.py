import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import yaml
class QNetwork(nn.Module):#输入一个S值，输出一个a值,
    def __init__(self,config):
        super().__init__()
        self.device=config['device']
        self.fc = nn.Sequential(
            nn.Linear(config['state_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, config['action_dim'])
        )
        self.to(self.device)  # 将网络移到指定设备

    def forward(self, x):
        x=x.to(self.device)
        return self.fc(x)


class DQNAgent:
    def __init__(self,config):
        self.q_net = QNetwork(config)       # 当前网络
        self.target_net = QNetwork(config)  # 目标网络
        self.target_net.load_state_dict(self.q_net.state_dict())  # 将目标网络和当前网络初始化一致，避免网络不一致导致的训练波动
        self.best_net = QNetwork(config)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config['learning_rate'])#1e-3
        self.replay_buffer = deque(maxlen=config['buffer_lenth']) #10000          # 经验回放缓冲区
        self.batch_size = config['batch_size']#64
        self.gamma = config['gamma']#0.99
        self.epsilon = config['epsilon_start']#1
        self.update_target_freq = config['update_frequence']#100  # 目标网络更新频率
        self.device=config['device']
        self.step_count = 0
        self.best_reward = 0
        self.best_avg_reward = 0
        self.eval_episodes = 5  # 评估时的episode数量
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)  # CartPole有2个动作（左/右）
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            return q_values.cpu().detach().numpy().argmax()
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从缓冲区随机采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算当前Q值
        #self.q_net(states):(B*action_dim),相当于给出每一个输入的输出所有动作的取值
        #actions是索引，表示这次行动取哪个action对应的值
        #actions.unsqueeze(1)形状为(B*1),那么gather之后的结果就是(B*1)
        #action中的索引为[(0,0),(1,0),...(B,0)]
        #然后dim=1,就是将actions中的值替换索引dim=1的位置，所以就变成
        #[(0,a0),(1,a1),...(B,aB)]然后将索引读取states中的状态就可以求得
        #[state_{0,a0},state_{1,a1},...state_{B,aB}],最后squeeze消除dim=1
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            #dones=True说明下一个状态不可达
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            # 使用深拷贝更新目标网络参数
            #{
            #   'layer1.weight': tensor(...),
            #   'layer1.bias': tensor(...),
            #   'layer2.weight': tensor(...),
            #   ...
            # }
            #nn,module中使用键值对存放参数，items返回键值对
            self.target_net.load_state_dict({
                k: v.clone() for k, v in self.q_net.state_dict().items()
            })
    def save_model(self, path="./output/best_model.pth"):
        if not os.path.exists("./output"):
            os.makedirs("./output")
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def evaluate(self, env):
        """评估当前模型的性能"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # 关闭探索
        total_rewards = []

        for _ in range(self.eval_episodes):
            state = env.reset()[0]
            episode_reward = 0
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if done or episode_reward > 2e4:
                    break
            total_rewards.append(episode_reward)

        self.epsilon = original_epsilon  # 恢复探索
        return np.mean(total_rewards)
    

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
config['state_dim']=state_dim
config['action_dim']=action_dim
agent = DQNAgent(config)
def train(config,env,agent):
    #每个 episode（回合）代表一次从环境开始（env.reset()）到结束（done=True）的完整游戏过程。
    for episode in range(config["episode"]):
        state = env.reset()[0]
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()

            total_reward += reward
            state = next_state
            if done or total_reward > 2e4:
                break

        # epsilon是探索系数，随着每一轮训练，epsilon 逐渐减小
        agent.epsilon = max(config["epsilon_end"], agent.epsilon * config["epsilon_decay"])  

        # 每10个episode评估一次模型
        if episode % 10 == 0:
            eval_env = gym.make('CartPole-v1')
            avg_reward = agent.evaluate(eval_env)
            eval_env.close()

            if avg_reward > agent.best_avg_reward:
                agent.best_avg_reward = avg_reward
                # 深拷贝当前最优模型的参数
                agent.best_net.load_state_dict({k: v.clone() for k, v in agent.q_net.state_dict().items()})
                agent.save_model(path=f"./output/best_model.pth")
                print(f"New best model saved with average reward: {avg_reward}")

        print(f"Episode: {episode}, Train Reward: {total_reward}, Best Eval Avg Reward: {agent.best_avg_reward}")
# train(config,env,agent)
# 测试并录制视频
agent.epsilon = 0  # 关闭探索策略
test_env = gym.make('CartPole-v1', render_mode='rgb_array')
test_env = RecordVideo(test_env, "./DQN_videos", episode_trigger=lambda x: True)  # 保存所有测试回合
best_model_path = "./DQN_output/best_model.pth"  # 你的保存路径
agent.q_net.load_state_dict(torch.load(best_model_path, map_location=config['device']))  # 加载到GPU或CPU
agent.q_net.eval()  # 设置为评估模式，关闭Dropout等
# agent.q_net.load_state_dict(agent.best_net.state_dict())  # 使用最佳模型
for episode in range(3):  # 录制3个测试回合
    state = test_env.reset()[0]
    total_reward = 0
    steps = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = test_env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

        # 限制每个episode最多1500步,约30秒,防止录制时间过长
        if done or steps >= 1500:
            break

    print(f"Test Episode: {episode}, Reward: {total_reward}")

test_env.close()