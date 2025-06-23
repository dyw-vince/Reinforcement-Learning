import gymnasium as gym
import torch
import numpy as np
from torchvision import transforms
from gymnasium.spaces import Box
from FrameStack import FrameStack
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
discrete_actions = [
    np.array([0.0, 0.0, 0.0]),  # no-op
    np.array([0.0, 0.8, 0.0]),  # gas
    np.array([0.0, 0.0, 0.8]),  # brake
    np.array([-0.2, 0.5, 0.0]),  # left + gas
    np.array([0.2, 0.5, 0.0]),   # right + gas
    np.array([-0.5, 0.0, 0.8]),  # left + brake
    np.array([0.5, 0.0, 0.8]),   # right + brake
    np.array([-0.5, 0.2, 0.0]),  # left + les_gas
    np.array([0.5, 0.2, 0.0]),   # right + les_gas
    np.array([-0.8, 0.0, 0.0]),  # left
    np.array([0.8, 0.0, 0.0]),   # right
]
#跳帧感知小车的位移量，每隔五帧作为小车的一次action
class CarV2SkipFrame(gym.Wrapper):
    def __init__(self, env, skip: int):
        """skip frame
        Args:
            env (_type_): _description_
            skip (int): skip frames
        """
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        reward_list = []
        done = False
        total_reward = 0
        for i in range(self.skip):
            obs, reward, done, info, _ = self.env.step(action)
            out_done = self.judge_out_of_route(obs)
            done = done or out_done
            reward = -5 if out_done else reward
            # reward = -100 if out_done else reward
            # reward = reward * 10 if reward > 0 else reward
            total_reward += reward
            reward_list.append(reward)
            if done:
                break
        return obs[:84, 6:90, :], total_reward, done, info, _
    
    def judge_out_of_route(self, obs):
        s = obs[:84, 6:90, :]
        #取第76行的13个像素的前两个点和后两个点的绿色通道，如果全部都是绿色那么就出界了
        out_sum = (s[75, 35:48, 1][:2] > 200).sum() + (s[75, 35:48, 1][-2:] > 200).sum()
        return out_sum == 4

    def reset(self, seed=0, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        # steering  gas  breaking
        action = discrete_actions[0]
        for i in range(45):
            obs, reward, done, info, _ = self.env.step(action)
        return obs[:84, 6:90, :], info

#CNN想要感知连续的动作，给他连续的4次动作作为CNN的一次输入
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip: int):
        """skip frame
        Args:
            env (_type_): _description_
            skip (int): skip frames
        """
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            obs, reward, done, info, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info, _

class PreprocessObservation(gym.ObservationWrapper):
    def __init__(self, env, shape: int):
        super().__init__(env)
        self.shape = (shape, shape)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),                  
            # 把原始图像（通常是 numpy.ndarray 格式，形状为 [H, W, C]）转换成 PIL.Image 格式，
            # 方便使用 Resize、Grayscale 等 PIL 操作。
            transforms.Grayscale(num_output_channels=1),
            #将 RGB 彩色图转换为灰度图，输出只有一个通道。
            transforms.Resize(self.shape),
            #将图像缩放为指定大小，例如 (84, 84)。
            transforms.ToTensor() 
            # 将 PIL.Image 转换为 PyTorch 的 Tensor，值归一化到 [0.0, 1.0]，并自动转换通道为 (C, H, W)。                                        # (1, H, W), float32 in [0,1]
        ])
        self.observation_space = Box(low=0.0, high=1.0, shape=(1, shape, shape), dtype=np.float32)

    def observation(self, obs):
        obs = self.transform(obs)  # returns tensor (1, H, W)
        return obs


class QNetwork(nn.Module):#输入一个图像，输出一个动作
    def __init__(self,config):
        super().__init__()
        #输入(f,H,w),f为隔多少帧作为一次,H=W=96
        self.device=config['device']
        self.conv=nn.Sequential(
            nn.Conv2d(config['frame_num'], 32, kernel_size=8, stride=4),  # (84->20)快速降低空间维度
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (20->9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (9->7)提取高层语义
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, config['action_dim'])
        )
        self.to(self.device)  # 将网络移到指定设备

    def forward(self, x):
        x=x.to(self.device)
        x=self.conv(x)
        x = x.view(x.size(0), -1)
        x=self.fc(x)
        return x
    

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
            if self.step_count<200:
                return random.randint(0, len(discrete_actions[:2])-1) 
            else:
                return random.randint(0, len(discrete_actions)-1) 
        else:
            state_tensor = torch.FloatTensor(np.stack(state)).squeeze(2)
            state_tensor = state_tensor.permute(1,0,2,3).to(self.device)
            q_values = self.q_net(state_tensor)
            return q_values.cpu().detach().numpy().argmax()#选择得到动作中的最大值
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    def judge_out_of_route(self,state):
        # 取车底部区域，适度加宽加高，提升鲁棒性
        region = state[70:80, 40:56, 1]  # G通道
        green_mask = region > 200
        green_ratio = green_mask.sum() / green_mask.size
        return green_ratio > 0.6  # 超过60%是草地则判出界
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # 从缓冲区随机采样
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.stack(states)).squeeze(2).to(self.device)
        actions = torch.tensor(np.stack(actions)).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.stack(rewards)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).squeeze(2).to(self.device)
        dones = torch.FloatTensor(np.stack(dones)).view(-1, 1).to(self.device)
        # states = torch.stack(states).float().to(self.device)
        # actions = torch.LongTensor(actions).to(self.device)
        # rewards = torch.FloatTensor(rewards).to(self.device)
        # next_states = torch.stack(next_states).float().to(self.device)
        # dones = torch.FloatTensor(dones).to(self.device)

        # 计算当前Q值
        #self.q_net(states):(B*action_dim),相当于给出每一个输入的输出所有动作的取值
        #actions是索引，表示这次行动取哪个action对应的值
        #actions.unsqueeze(1)形状为(B*1),那么gather之后的结果就是(B*1)
        #action中的索引为[(0,0),(1,0),...(B,0)]
        #然后dim=1,就是将actions中的值替换索引dim=1的位置，所以就变成
        #[(0,a0),(1,a1),...(B,aB)]然后将索引读取states中的状态就可以求得
        #[state_{0,a0},state_{1,a1},...state_{B,aB}],最后squeeze消除dim=1
        current_q = self.q_net(states).gather(1, actions).squeeze()
        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            #dones=True说明下一个状态不可达
            target_q = rewards + self.gamma * next_q * (1 - dones)
        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q, target_q.squeeze(1))
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
            step=0
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = env.step(discrete_actions[action])
                episode_reward += reward
                state = next_state
                step+=1
                if done or episode_reward > 1000 or step>750:
                    break
            total_rewards.append(episode_reward)

        self.epsilon = original_epsilon  # 恢复探索
        return np.mean(total_rewards)


def train(config,env,agent):
    #每个 episode（回合）代表一次从环境开始（env.reset()）到结束（done=True）的完整游戏过程。
    max_steps = 1000
    for episode in range(config["episode"]):
        # stacked_states = FrameStack(env, num_stack=5)
        state = env.reset()[0]
        total_reward = 0
        while True:
            action = agent.choose_action(state)#映射到离散空间
            next_state, reward,done,_,_= env.step(discrete_actions[action])
            agent.store_experience(state, action, reward, next_state, done)
            # stacked_states = FrameStack(next_state, num_stack=5)
            agent.train()
            total_reward += reward
            state=next_state
            if done or total_reward > 1000 :
                break
        # epsilon是探索系数，随着每一轮训练，epsilon 逐渐减小
        agent.epsilon = max(config["epsilon_end"], agent.epsilon * config["epsilon_decay"])  

        # 每10个episode评估一次模型
        if (episode) % 10 == 0:
            eval_env = gym.make('CarRacing-v3',continuous=True)
            eval_env = FrameStack(PreprocessObservation(CarV2SkipFrame(eval_env, skip=SKIP_N), shape=84), num_stack=STACK_N)
            avg_reward = agent.evaluate(eval_env)
            eval_env.close()

            if avg_reward > agent.best_avg_reward:
                agent.best_avg_reward = avg_reward
                # 深拷贝当前最优模型的参数
                agent.best_net.load_state_dict({k: v.clone() for k, v in agent.q_net.state_dict().items()})
                agent.save_model(path=f"./output/best_model.pth")
                print(f"New best model saved with average reward: {avg_reward}")

        print(f"Episode: {episode}, Train Reward: {total_reward}, Best Eval Avg Reward: {agent.best_avg_reward}")

def eval_and_record(agent):
    agent.epsilon = 0  # 关闭探索策略
    test_env = gym.make("CarRacing-v3",render_mode="rgb_array")
    test_env = PreprocessObservation(CarV2SkipFrame(test_env, skip=SKIP_N), shape=84)
    test_env = FrameStack(RecordVideo(test_env, "./DQN_videos", episode_trigger=lambda x: True), num_stack=STACK_N)  # 保存所有测试回合
    best_model_path = "./output/best_model.pth"  # 你的保存路径
    agent.q_net.load_state_dict(torch.load(best_model_path, map_location=config['device']))  # 加载到GPU或CPU
    agent.q_net.eval()  # 设置为评估模式，关闭Dropout等
    # agent.q_net.load_state_dict(agent.best_net.state_dict())  # 使用最佳模型
    for episode in range(3):  # 录制3个测试回合
        state = test_env.reset()[0]
        steps = 0
        total_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = test_env.step(discrete_actions[action])
            total_reward += reward
            state=next_state
            steps += 1

            # 限制每个episode最多1500步,约30秒,防止录制时间过长
            if done or steps >= 1000:
                break
        print(f"Test Episode: {episode}, Reward: {total_reward}")
    test_env.close()


#######main########
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
env_name = 'CarRacing-v3'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
config['state_dim']=state_dim
config['action_dim']=len(discrete_actions)
SKIP_N = 5
STACK_N = 4
env = FrameStack(PreprocessObservation(CarV2SkipFrame(env, skip=SKIP_N), shape=84), num_stack=STACK_N)
agent = DQNAgent(config)
train(config,env,agent)
eval_and_record(agent)
# 测试并录制视频
