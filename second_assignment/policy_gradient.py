import gym
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
class ActorNet(nn.Module):#输入一个S值，输出所有a值的概率,
    def __init__(self,config):
        super().__init__()
        self.device=config['device']
        self.fc = nn.Sequential(
            nn.Linear(config['state_dim'], config['embed_dim']),
            nn.ReLU(),
            nn.Linear(config['embed_dim'], config['embed_dim']),
            nn.ReLU(),
            nn.Linear(config['embed_dim'], config['action_dim'])
        )
        self.to(self.device)  # 将网络移到指定设备

    def forward(self, x):#(B,a)
        x=x.to(self.device)
        x=self.fc(x)
        return F.softmax(x,dim=1)
class CriticNet(nn.Module):#输入一个S值，输出一个价值
    def __init__(self,config):
        super().__init__()
        self.device=config['device']
        self.fc = nn.Sequential(
            nn.Linear(config['state_dim'], config['embed_dim']),
            nn.ReLU(),
            nn.Linear(config['embed_dim'], config['embed_dim']),
            nn.ReLU(),
            nn.Linear(config['embed_dim'], 1)
        )
        self.to(self.device)  # 将网络移到指定设备

    def forward(self, x):
        x=x.to(self.device)
        return self.fc(x)

class PGAgent(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.gamma = config['gamma']
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.device=config['device']
        self.actor=ActorNet(config).to(self.device)
        self.critic=CriticNet(config).to(self.device)
        self.best_actor=ActorNet(config).to(self.device)
        self.best_critic=CriticNet(config).to(self.device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),self.actor_lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),self.critic_lr)
        self.log_a = []
        self.ep_r = []
        self.best_reward = 0
        self.best_avg_reward = 0
        self.eval_episodes = 5  # 评估时的episode数量
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)           # [B,state_dim]->[1,state_dim]
        probs = self.actor(state)                                 # [B,action_dim]
        action_dist = torch.distributions.Categorical(probs)                # 根据概率采样
        action = action_dist.sample()  # 按照probs定义的概率随机采样一个动作 #[B,1]
        return action.item()

    def save_model(self, path_actor,path_critic):
        if not os.path.exists("./PG_output"):
            os.makedirs("./PG_output")
        torch.save(self.best_actor.state_dict(), path_actor)
        torch.save(self.best_critic.state_dict(), path_critic)
        print(f"Model saved to {path_actor} and {path_critic}")
    def train(self,states,actions,rewards,next_states,dones):
        states=torch.tensor(states,dtype=torch.float).to(self.device)#(B,states_dim)
        actions=torch.tensor(actions,dtype=torch.long).unsqueeze(1).to(self.device)#(B,1)
        rewards=torch.tensor(rewards,dtype=torch.float).unsqueeze(1).to(self.device)#(B,1)
        next_states=torch.tensor(next_states,dtype=torch.float).to(self.device)#(B,4)
        dones=torch.tensor(dones,dtype=torch.float).unsqueeze(1).to(self.device)#(B,1)
        # print(f"states_shape:{states.shape},actions_shape:{actions.shape},rewards_shape:{rewards.shape},next_states_shape:{next_states.shape},dones_shape:{dones.shape}")
        td_target=rewards + self.gamma * self.critic(next_states) * (1-dones)
        td_delta=td_target-self.critic(states)
        log_probs = torch.log(self.actor(states).gather(1, actions))
        #actor学习一个更好的动作，critic使得估计的价值更准
        actor_loss= actor_loss = torch.mean(-(log_probs * td_delta.detach()))
        critic_loss = torch.mean(torch.nn.functional.mse_loss(self.critic(states), td_target.detach()))
        #detach防止作为参数反向传播，因为td_target是critic的信息，不希望他影响自己
        #外面还要使用torch_mean是因为mse_loss得到的是一个[B,1]的平方损失
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
    def evaluate(self,env,greedy=None):
        total_reward = 0.0
        for _ in range(self.eval_episodes):
            state = env.reset()[0]
            done = False
            episode_return = 0.0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    if greedy:
                        # 取actor输出最大概率动作
                        probs = self.actor(state_tensor)
                        action = torch.argmax(probs, dim=-1).item()
                    else:
                        # 仍然根据分布采样（如果想要）
                        probs = self.actor(state_tensor)
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample().item()
                next_state, reward, done, _, _ = env.step(action)
                episode_return += reward
                state = next_state
            total_reward += episode_return
        avg_reward = total_reward / self.eval_episodes
        return avg_reward
if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    env = gym.make('CartPole-v1')
    config['action_dim'] = env.action_space.n
    config['state_dim'] = env.observation_space.shape[0]
    agent = PGAgent(config)

    x=[]
    total_rewards=[]
    num_round=4
    for num in range(num_round):
        for episode in range(config['episode']):
            print(f"round{num+1}:开始第{episode}轮训练")
            state = env.reset()[0]
            episode_reward = 0
            states=[]
            actions=[]
            next_states=[]
            rewards=[]
            dones=[]
            while True:
                action = agent.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                episode_reward += reward
                state = next_state
                if done or episode_reward > 1500:
                    break
            print('Episode {:03d} | Reward:{:.03f}'.format(episode, episode_reward))
            agent.train(states,actions,rewards,next_states,dones)
            x.append(episode+1)
            total_rewards.append(episode_reward)
            # 每10个episode评估一次模型
            if episode % 20 == 0:
                print(f"第{episode//20+1}次评估")
                eval_env = gym.make('CartPole-v1')
                avg_reward = agent.evaluate(eval_env)
                eval_env.close()

                if avg_reward > agent.best_avg_reward:
                    agent.best_avg_reward = avg_reward
                    # 深拷贝当前最优模型的参数
                    agent.best_actor.load_state_dict({k: v.clone() for k, v in agent.actor.state_dict().items()})
                    agent.best_critic.load_state_dict({k: v.clone() for k, v in agent.critic.state_dict().items()})
                    agent.save_model(path_actor=f"./PG_output/best_actor.pth",path_critic=f"./PG_output/best_critic.pth")
                    agent.best_avg_reward = avg_reward
                    print(f"New best model saved with average reward: {avg_reward}")
        plt.plot(x, total_rewards)
        plt.show()
# 测试并录制视频
agent.epsilon = 0  # 关闭探索策略
test_env = gym.make('CartPole-v1', render_mode='rgb_array')
test_env = RecordVideo(test_env, "./PG_videos", episode_trigger=lambda x: True)  # 保存所有测试回合
best_actor_path = "./PG_output/best_actor.pth"  
best_critic_path = "./PG_output/best_critic.pth"
agent.actor.load_state_dict(torch.load(best_actor_path, map_location=config['device'],weights_only=True))  # 加载到GPU或CPU
agent.critic.load_state_dict(torch.load(best_critic_path, map_location=config['device'],weights_only=True))  # 加载到GPU或CPU
agent.actor.eval()  
agent.critic.eval() # 设置为评估模式，关闭Dropout等
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