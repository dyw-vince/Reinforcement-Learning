import matplotlib.pyplot as plt
from my_QMIX import *
import numpy as np
from utils import *
import yaml
import wandb
class Training:
    def __init__(self,config,test):
        self.config=config
        if test:
            self.env = make_env(self.config['max_episode_times'],render_mode='rgb_array')
        else:
            self.env = make_env(self.config['max_episode_times'],render_mode=None) # Discrete action space
        self.config['N'] = self.env.max_num_agents  # The number of agents
        self.config['action_dim_n'] = [self.env.action_spaces[agent].n for agent in self.env.agents]  
        # actions dimensions of N agents
        self.config['action_dim'] = self.config['action_dim_n'][0]  
        # The dimensions of an agent's action space
        self.config['state_dim_n'] = [self.env.observation_spaces[agent].shape[0] for agent in self.env.agents]
        # obs dimensions of N agents
        self.config['state_dim'] = self.config['state_dim_n'][0]  
        # The dimensions of an agent's observation space
        self.config['global_state_dim'] = np.sum(self.config['state_dim_n']) 
         # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        self.agent_n = QMIX_Agents(self.config)
        self.agent_names= [agent for agent in self.env.agents]
        self.buffer = Buffer(self.config)
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.config['epsilon_decay']=(config['epsilon'] - config['epsilon_min']) / config['epsilon_decay_steps']
        self.train_log_intervals=config['train_log_intervals']
        self.total_steps = 0
        if self.config['use_reward_norm']:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.config['N'])
        wandb.init(
            project="qmix_simple_spread",  # 你的项目名
            config=config  # 可选，记录你的超参数设置
        )
    def run(self):
        while self.total_steps < self.config['max_train_steps']:
            if self.total_steps % self.config['evaluate_freq'] == 0:
                self.evaluate()  # Evaluate the policy every 'evaluate_freq' steps

            episode_rewards, episode_steps = self.run_episode(evaluate=False)  # Run an episode
            if self.total_steps % self.train_log_intervals ==0:
                print(f"第{self.total_steps}轮训练的rewards:{episode_rewards:.2f}")
            self.total_steps += episode_steps

            if self.buffer.episode_num>=self.config['batch_size']:
                losses=[]
                for i in range(self.config['train_times_per_episode']):
                    loss=self.agent_n.train(self.buffer)  # Training
                    losses.append(loss)
                wandb.log({
                    "loss": sum(losses) / len(losses),
                }, step=self.total_steps)
        self.evaluate()
        self.env.close()

    def run_episode(self,evaluate):
        episode_reward = 0
        states,_ = self.env.reset()
        self.agent_n.current_Q_RNN.init_hidden(self.config['N']) #将RNN的隐变量初始化
        states_n = dict2np(states)
        onehot_actions_n = np.zeros((self.config['N'], self.config['action_dim']))
        for episode_step in range(self.config['max_episode_times']):#25
            epsilon=0 if evaluate else self.agent_n.epsilon
            actions_n, actions_dict,onehot_actions_n = self.agent_n.choose_action(self.agent_names,states_n,onehot_actions_n,epsilon) 
            global_state = np.array(states_n).flatten()
            # Get actions and the corresponding log probabilities of N agents
            next_states_n, rewards_n, dones_n, _, _ = self.env.step(actions_dict)
            dones_n = dict2np(dones_n)
            rewards_n= dict2np(rewards_n)
            episode_reward += np.mean(rewards_n)
            if not evaluate:
                if self.config['use_reward_norm']:
                    rewards_n = self.reward_norm(rewards_n)
                # Store the transition
                self.buffer.store(episode_step, states_n, global_state,onehot_actions_n, actions_n, rewards_n, dones_n)
            states_n = dict2np(next_states_n)
            self.agent_n.epsilon = self.agent_n.epsilon - self.config['epsilon_decay'] if self.agent_n.epsilon - self.config['epsilon_decay'] > self.config['epsilon_min'] else  self.config['epsilon_min']
            if all(dones_n):
                break
        #在每个episode的最后一次还需要再生成一个value作为末尾，否则后向视角无法传播
        if not evaluate:
            global_state = np.array(states_n).flatten()
            self.buffer.store_last_state(episode_step + 1,states_n, global_state)

        return episode_reward, episode_step + 1
    
    def evaluate(self):
        evaluate_reward = 0
        for _ in range(self.config['evaluate_times']):
            episode_reward, _= self.run_episode(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.config['evaluate_times']
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        wandb.log({"evaluate_reward": evaluate_reward}, step=self.total_steps)
        # Save the rewards and models
        os.makedirs('./data_train', exist_ok=True)
        np.save('./data_train/QMIX_rewards.npy',np.array(self.evaluate_rewards))
        self.agent_n.save_model(self.total_steps,evaluate_reward)


