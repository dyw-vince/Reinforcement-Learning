import gym
import imageio
import torch
import numpy as np
from DDPG import AgentDDPG  # 替换成你的 Agent 类路径
env = gym.make("CarRacing-v2", render_mode="rgb_array")
video = []
agent = AgentDDPG(env.action_space, model_outputs=2)
agent.load_model(path="best_models/")
state, _ = env.reset()
done = False

while not done:
    frame = env.render()
    video.append(frame)

    action, train_action = agent.get_action(state, add_noise=False)  # 不加噪声，测试稳定策略
    action /= 4
    state, reward, done, truncated, _ = env.step(action)
    reward = np.clip(reward, a_max=1, a_min=-10)
    if truncated: break
env.close()
imageio.mimsave("ddpg_best_model_run.mp4", video, fps=30)