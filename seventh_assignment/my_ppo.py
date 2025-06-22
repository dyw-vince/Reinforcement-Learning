import os
import re
import torch
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import *
from utils import *
# Hyperparameters

class MAPPO_Agents:
    def __init__(self,config):
        self.action_dim=config['action_dim']
        self.batch_size=config['batch_size']
        self.best_store_nums=config['best_store_nums']
        self.best_models=[]
        self.clip_epsilon=config['clip_epsilon']#epsilon截断大小
        self.config=config
        self.device=config['device']
        self.entropy_coef=config['entropy_coef']#0.01
        self.global_state_dim=config['global_state_dim']
        self.gamma=config['gamma']#0.99
        self.Lambda=config['lambda']#0.95
        self.max_eposide_times=config['max_episode_times']#每一episode的最大步数
        self.max_train_steps=config['max_train_steps']
        self.mini_batch_size=config['mini_batch_size']
        self.N=config['agent_num']
        self.state_dim=config['state_dim']
        self.train_times_per_episode=config['train_times_per_episode']
        self.update_epoches=config['update_epoches']#在training的一个epoch中更新网络的轮数
        self.use_adv_norm = config['use_adv_norm']
        self.use_grad_clip=config['use_grad_clip']

        self.actor=Actor(config,self.state_dim).to(self.device)
        self.critic=Critic(config,self.global_state_dim).to(self.device)
        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer=torch.optim.Adam(self.ac_parameters,lr=config['learning_rate'])#1e-3
        self.scheduler= CosineAnnealingLR(self.ac_optimizer, T_max=self.max_train_steps, eta_min=0)
        
    def choose_action(self, states_n,names,evaluate):
        with torch.no_grad():
            actor_inputs = torch.tensor(states_n, dtype=torch.float32).to(self.device)  # obs_n.shape=(N，obs_dim)
            probs = self.actor(actor_inputs)  # prob.shape=(N,action_dim)
            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                actions_n = probs.argmax(dim=-1)
                actions_n= actions_n.cpu().numpy()
                actions_dict = {}
                for i, agent in enumerate(names):
                    actions_dict[agent] = actions_n[i]
                return None,actions_dict,None
            else:
                dist = Categorical(probs=probs)
                actions_n = dist.sample()
                logprobs_n = dist.log_prob(actions_n)
                actions_n=actions_n.cpu().numpy()
                logprobs_n=logprobs_n.cpu().numpy()
                actions_dict = {}
                for i, agent in enumerate(names):
                    actions_dict[agent] = actions_n[i]
                return actions_n, actions_dict,logprobs_n
    def get_value(self, global_state):
        #return [N,]
        with torch.no_grad():
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            critic_inputs = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1).to(self.device)  # (state_dim,)-->(N,state_dim)
            values_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return values_n.cpu().numpy().flatten()
    def train(self, buffer):
        batch = buffer.get_training_data()  # get training data
        # Calculate the advantage using GAE
        advantages,vs=advantage_and_V(batch,self.config)
        advantages=advantages.to(self.device)
        vs=vs.to(self.device)
        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = get_inputs(batch,self.N,self.device)
        # Optimize policy for K epochs:
        for _ in range(self.train_times_per_episode):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                #一个batch为32个episode,每次用8个episode更新
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size,max_eposide_times, N, action_dim)
                    values_now.shape=(mini_batch_size,max_eposide_times, N)
                """
                new_probs = self.actor(actor_inputs[index])
                new_values = self.critic(critic_inputs[index])
                dist_now = Categorical(new_probs)
                entropy = dist_now.entropy()  # dist_entropy.shape=(batch_size,max_eposide_times, N)
                # batch['a_n'][index].shape=(mini_batch_size,max_eposide_times, N)
                new_logprobs = dist_now.log_prob(batch['actions_n'][index].to(self.device))#生成和action_n一样形状的分布
                # a_logprob_n_now.shape=(batch_size,max_eposide_times, N)
                ratios = torch.exp(new_logprobs - batch['logprobs_n'][index].to(self.device).detach())  # ratios.shape=(batch_size, max_eposide_times, N)
                #本身是除法，我们使用P1/P2=exp(logP1-logP2)更稳定
                surr1 = ratios * advantages[index]
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy
                critic_loss = (new_values - vs[index]) ** 2
                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()
                self.scheduler.step()

    def save_model(self,steps,reward):
        actor_path=f"./model/MAPPO_actor_step_{int(steps/1000)}k_reward:{reward:.2f}.pth"
        critic_path=f"./model/MAPPO_critic_step_{int(steps/1000)}k_reward:{reward:.2f}.pth"
        # 如果当前还没满5个，直接保存
        if len(self.best_models) < self.best_store_nums:
            torch.save(self.actor.state_dict(),actor_path)
            torch.save(self.critic.state_dict(),critic_path)
            self.best_models.append((reward, actor_path,critic_path))

        else:
            # 找出最差的模型
            min_reward, min_actor_path, min_critic_path = min(self.best_models, key=lambda x: x[0])
            if reward > min_reward:
                # 保存新模型
                torch.save(self.actor.state_dict(),actor_path)
                torch.save(self.critic.state_dict(),critic_path)
                # 删除旧模型文件
                if os.path.exists( min_actor_path):
                    os.remove(min_actor_path)
                if os.path.exists(min_critic_path ):
                    os.remove(min_critic_path )
                # 替换最差模型
                self.best_models.remove((min_reward, min_actor_path, min_critic_path))
                self.best_models.append((reward, actor_path,critic_path))
        
    def load_model(self,model_dir="./model"):
        # 找出所有以 MAPPO_actor_step 开头的模型文件
        model_files = [f for f in os.listdir(model_dir) if f.startswith("MAPPO_actor_step") and f.endswith(".pth")]

        best_reward = -float('inf')
        best_model_path = None

        for file in model_files:
            # 使用正则提取 reward 数字（支持小数,负数）
            match = re.search(r"reward:(-?[0-9.]+)", file)
            if match:
                reward = float(match.group(1).rstrip('.') )
                if reward > best_reward:
                    best_reward = reward
                    best_model_path = os.path.join(model_dir, file)

        if best_model_path:
            print(f"Loading best model from {best_model_path} with reward {best_reward}")
            self.actor.load_state_dict(torch.load(best_model_path))
        else:
            print("No valid model file found.")
    