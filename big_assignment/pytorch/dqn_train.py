import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import cv2
import os
import csv
import pygame
from collections import deque
from flappy_bird_env import FlappyBirdEnv

# === 超参数 ===
GAMMA = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 0.1       # ✅ 初始值
EPSILON_END = 0.0001      # ✅ 最小值
EPSILON_DECAY_EPISODES = 2000  # ✅ 线性衰减总回合数
TARGET_UPDATE_FREQ = 10

# === 路径 ===
SAVE_PATH = 'dqn_flappy_bird_best.pth'
FINAL_MODEL_PATH = 'dqn_final_model.pth'
TRAIN_LOG_PATH = 'train_log.csv'
VIDEO_PATH = 'demo_video.mp4'

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        next_q_values[dones] = 0.0
        expected_q_values = rewards + GAMMA * next_q_values

        loss = self.loss_fn(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def save_training_log(log_data, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'TotalReward', 'Score', 'Epsilon'])
        writer.writerows(log_data)

def record_demo(agent, env, path='demo_video.mp4'):
    state = env.reset()
    done = False
    frames = []

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, score = env.step(action)
        state = next_state
        env.render()

        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(frame)

    height, width, _ = frames[0].shape
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for f in frames:
        video.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    video.release()
    print(f"✅ 已保存演示视频: {path}")

# === 线性 epsilon 衰减函数 ===
def linear_decay(episode):
    frac = min(1.0, episode / EPSILON_DECAY_EPISODES)
    return EPSILON_START - frac * (EPSILON_START - EPSILON_END)

def main():
    env = FlappyBirdEnv()
    agent = DQNAgent(state_dim=env.state_space_dim, action_dim=len(env.action_space))
    max_score = 0
    training_log = []

    for episode in range(1, EPSILON_DECAY_EPISODES + 1):
        agent.epsilon = linear_decay(episode)  # ✅ 每回合更新 epsilon

        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, score = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            if len(agent.memory) >= BATCH_SIZE:
                agent.learn()

            state = next_state
            total_reward += reward

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if score > max_score:
            max_score = score
            torch.save(agent.policy_net.state_dict(), SAVE_PATH)

        print(f"回合: {episode}, 奖励: {total_reward:.2f}, 分数: {score}, Epsilon: {agent.epsilon:.6f}")
        training_log.append([episode, total_reward, score, agent.epsilon])

    torch.save(agent.policy_net.state_dict(), FINAL_MODEL_PATH)
    save_training_log(training_log, TRAIN_LOG_PATH)
    print(f"✅ 训练日志保存到: {TRAIN_LOG_PATH}")

    agent.policy_net.load_state_dict(torch.load(SAVE_PATH))
    agent.epsilon = 0.0
    record_demo(agent, env, VIDEO_PATH)

    try:
        pygame.quit()
    except:
        pass

if __name__ == '__main__':
    main()
