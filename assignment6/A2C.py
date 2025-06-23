import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np

class OneDNavigationEnv(gym.Env):
    def __init__(self):
        super(OneDNavigationEnv, self).__init__()
        self.observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(2)
        self.reset()

    def reset(self):
        self.position = 0
        self.steps = 0
        return self.position

    def step(self, action):
        self.steps += 1

        if action == 0 and self.position > 0:
            self.position -= 1
        elif action == 1 and self.position < 9:
            self.position += 1

        reward = -1
        done = False

        if self.position == 5:
            reward = -50
        if self.position == 9:
            reward = 100
            done = True
        if self.steps >= 20:
            done = True

        return self.position, reward, done, {}

    def render(self, mode='human'):
        print(f"Agent Position: {self.position}")


class A2CNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A2CNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

    def act(self, state):
        logits, _ = self.forward(state)
        prob = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_actions(self, state, action):
        logits, value = self.forward(state)
        prob = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(prob)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

def one_hot(state, num_classes=10):
    vec = np.zeros(num_classes)
    vec[state] = 1.0
    return torch.tensor(vec, dtype=torch.float32)

def train_a2c():
    env = OneDNavigationEnv()
    model = A2CNet(state_dim=10, action_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    gamma = 0.99
    entropy_beta = 0.01

    for episode in range(500):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        entropies = []

        for step in range(20):
            state_tensor = one_hot(state)
            action, log_prob = model.act(state_tensor)
            _, entropy, value = model.evaluate_actions(state_tensor, torch.tensor(action))
            
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            entropies.append(entropy)

            state = next_state
            if done:
                break

        R = 0 if done and state != 9 else 100
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.stack(returns)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        entropy_loss = -entropies.mean()

        loss = actor_loss + 0.5 * critic_loss + entropy_beta * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards).item():.2f}")

if __name__ == "__main__":
    train_a2c()