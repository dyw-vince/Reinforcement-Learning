import os
import re
import torch
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import *
from utils import *
# Hyperparameters

class QMIX_Agents:
    def __init__(self,config):
        self.action_dim=config['action_dim']
        self.add_agent_id=config['add_agent_id']
        self.add_last_action=config['add_last_action']
        self.batch_size=config['batch_size']
        self.best_store_nums=config['best_store_nums']
        self.best_models=[]
        self.config=config
        self.device=config['device']
        self.epsilon=config['epsilon'] #探索概率
        self.global_state_dim=config['global_state_dim']
        self.gamma=config['gamma']#0.99
        self.Lambda=config['lambda']#0.95
        self.max_eposide_times=config['max_episode_times']#每一episode的最大步数
        self.max_train_steps=config['max_train_steps']
        self.N=config['agent_num']
        self.state_dim=config['state_dim']
        self.target_update_frequence=config['target_update_frequence']
        self.tau=config['target_update_theta']
        self.train_step=0
        self.train_times_per_episode=config['train_times_per_episode']
        self.update_epoches=config['update_epoches']#在training的一个epoch中更新网络的轮数
        self.use_adv_norm = config['use_adv_norm']
        self.use_hard_update=config['use_hard_update']
        self.use_grad_clip=config['use_grad_clip']
        self.use_rnn=config['use_rnn']

        self.input_dim = self.state_dim
        if self.add_last_action:
            self.input_dim+=self.action_dim
        if self.add_agent_id:
            self.input_dim+=self.N
        if self.use_rnn:
            self.current_Q_RNN=QNet_RNN(config,self.input_dim).to(self.device)
            self.target_Q_RNN=QNet_RNN(config,self.input_dim).to(self.device)
        
        self.target_Q_RNN.load_state_dict(self.current_Q_RNN.state_dict())

        self.current_Q_MIX=Qmix(config).to(self.device)
        self.target_Q_MIX=Qmix(config).to(self.device)
        self.target_Q_MIX.load_state_dict(self.current_Q_MIX.state_dict())

        self.QMIX_parameters = list(self.current_Q_RNN.parameters()) + list(self.current_Q_MIX.parameters())
        self.optimizer=torch.optim.RMSprop(
            params=self.QMIX_parameters,
            lr=config['learning_rate'],
            alpha=0.99,
            eps=0.00001,
            weight_decay=1e-5
        )
       
    def choose_action(self, names,states_n, onehot_actions_n,epsilon):
        with torch.no_grad():
            if np.random.uniform() < epsilon:  # epsilon-greedy
                # Only available actions can be chosen
                actions_n = np.array([np.random.randint(self.action_dim) for _ in range(self.N)])
            else:
                inputs = []
                states_n = torch.tensor(states_n, dtype=torch.float32)  # states_n.shape=(N，states_dim)
                inputs.append(states_n)
                if self.add_last_action:
                    last_actions_n = torch.tensor(onehot_actions_n, dtype=torch.float32)
                    inputs.append(last_actions_n)
                if self.add_agent_id:
                    inputs.append(torch.eye(self.N))

                inputs = torch.cat([x for x in inputs], dim=-1).to(self.device)  # inputs.shape=(N,inputs_dim)
                q_value = self.current_Q_RNN(inputs)  # q_value.shape=(N,action_dim)
                actions_n = q_value.argmax(dim=-1).cpu().numpy() # actions_n.shape=(N,1)
                onehot_actions_n = np.eye(self.action_dim)[actions_n] #onehot_actions_n=(N,action_dim)
                #actions_n=[3,1,4]
                #onehot_actions_n=[np.array(0,0,1,0,0),
                #                  np.array(1,0,0,0,0),
                #                  np.array(0,0,0,1,0)]
            actions_dict = {}
            for i, agent in enumerate(names):
                actions_dict[agent] = actions_n[i]
                
        return actions_n, actions_dict,onehot_actions_n
        
    def train(self, buffer):
        self.train_step += 1
        batch = buffer.get_training_data()  # get training data
        # Calculate the advantage using GAE
        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        inputs= get_inputs(batch,self.config)#inputs=states + last_onehot_actions + agent_ids
        #inputs.shape=(batch_size,max_episode_len+1,N,input_dim)
        if self.use_rnn:
            self.current_Q_RNN.init_hidden(self.batch_size*self.N)
            self.target_Q_RNN.init_hidden(self.batch_size*self.N)
        q_currents, q_targets = [], []
        if self.use_rnn:
            for t in range(self.max_eposide_times):  # t=0,1,2,...(episode_len-1)
                q_current = self.current_Q_RNN(inputs[:, t].reshape(-1, self.input_dim))  
                # q_current.shape=(batch_size*N,action_dim)
                q_target = self.target_Q_RNN(inputs[:, t + 1].reshape(-1, self.input_dim))
                # q_target.shape=(batch_size*N,action_dim)
                q_currents.append(q_current.reshape(self.batch_size, self.N, -1))  
                q_targets.append(q_target.reshape(self.batch_size, self.N, -1))
                # q_currents[i].shape=(batch_size,N,action_dim)
            # Stack them according to the time (dim=1)
            q_currents = torch.stack(q_currents, dim=1).to(self.device) 
            # q_currents.shape=(batch_size,max_episode_len,N,action_dim) 
            q_targets = torch.stack(q_targets, dim=1).to(self.device) 
        else:
            q_currents = self.current_Q_RNN(inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
            q_targets = self.target_Q_RNN(inputs[:, 1:])
        r"""
        double Q learning:
        对于一个智能体而言：
        为了防止过大估计不能直接将batch中的actions_n输给Q_target,而是先将actions_n和states_n
        输入Q_current后,得到当前Q值最大的动作a*,再将a*输入Q_target
        Q_t=Q_current(s,a)->argmax_a Q_t= a*
        从current_Q_RNN中估计在t时刻,s状态,a动作下得到的t+1时刻的值Q_t和t+1时刻的最优的动作a',
        再从target_Q_RNN中得到t+1时刻,s'状态,a*动作下的下的值Q_{t+1}
        更新current_Q:
        a*=argmax_a Q_current(s,a)
        Q_current= Q_mix(\sum Q_current(s,a))
        Q_target= r+gamma*(1-dones_n)*Q_mix(\sum Q_target(s',a*))
        loss=[Q_target(s,a)-Q_current(s',a*)]^2
        """
        with torch.no_grad():
            q_last = self.current_Q_RNN(inputs[:, -1].reshape(-1, self.input_dim)).reshape(self.batch_size, 1, self.N, -1)
            q_nexts = torch.cat([q_currents[:, 1:], q_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,N,action_dim)
            a_argmax = torch.argmax(q_nexts, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, N, 1)
            q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len, N)
        # batch['actions_n'].shape(batch_size,max_episode_len, N)
        q_currents = torch.gather(q_currents, dim=-1, index=batch['actions_n'].unsqueeze(-1)).squeeze(-1)  
        # q_currents .shape(batch_size, max_episode_len, N)

        # Compute q_total using QMIX or VDN, q_total.shape=(batch_size, max_episode_len, 1)
        q_total_current = self.current_Q_MIX(q_currents, batch['global_states'][:, :-1])
        with torch.no_grad():
            q_total_target = self.target_Q_MIX(q_targets, batch['global_states'][:, 1:])

        targets =torch.mean(batch['rewards_n'],dim=-1, keepdim=True) + self.gamma * (1 - batch['dones_n'].all(dim=-1, keepdim=True).float()) * q_total_target
        loss =nn.MSELoss()(q_total_current,targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip: 
            torch.nn.utils.clip_grad_norm_(self.QMIX_parameters, 4.0)
        self.optimizer.step()
        # self.scheduler.step()
        if self.use_hard_update:
            if self.train_step % self.target_update_frequence == 0:
                self.target_Q_RNN.load_state_dict(self.current_Q_RNN.state_dict())
                self.target_Q_MIX.load_state_dict(self.current_Q_MIX.state_dict())
        else:
            # Softly update the target networks
            for param, target_param in zip(self.current_Q_RNN.parameters(), self.target_Q_RNN.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.current_Q_MIX.parameters(), self.target_Q_MIX.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return loss
    def save_model(self,steps,reward):

        os.makedirs(f"model", exist_ok=True)
        Q_RNN_path=f"./model/Q_RNN_step:{int(steps/1000)}k_reward:{reward:.2f}.pth"
        Q_MIX_path=f"./model/Q_MIX_step:{int(steps/1000)}k_reward:{reward:.2f}.pth"
        # 如果当前还没满5个，直接保存
        if len(self.best_models) < self.best_store_nums:
            torch.save(self.current_Q_RNN.state_dict(),Q_RNN_path)
            torch.save(self.current_Q_MIX.state_dict(),Q_MIX_path)
            self.best_models.append((reward, Q_RNN_path,Q_MIX_path))

        else:
            # 找出最差的模型
            min_reward, min_Q_RNN_path, min_Q_MIX_path = min(self.best_models, key=lambda x: x[0])
            if reward > min_reward:
                # 保存新模型
                torch.save(self.current_Q_RNN.state_dict(),Q_RNN_path)
                torch.save(self.current_Q_MIX.state_dict(),Q_MIX_path)
                # 删除旧模型文件
                if os.path.exists(min_Q_RNN_path):
                    os.remove(min_Q_RNN_path)
                if os.path.exists(min_Q_MIX_path):
                    os.remove(min_Q_MIX_path)
                # 替换最差模型
                self.best_models.remove((min_reward, min_Q_RNN_path,min_Q_MIX_path))
                self.best_models.append((reward, Q_RNN_path,Q_MIX_path))
        
    def load_model(self):
        # 找出所有以 MAPPO_actor_step 开头的模型文件
        best_Q_RNN_path=self.load_single_model(start_string="Q_RNN_step")
        best_Q_MIX_path=self.load_single_model(start_string="Q_MIX_step")
        self.current_Q_RNN.load_state_dict(torch.load(best_Q_RNN_path))
        self.current_Q_MIX.load_state_dict(torch.load(best_Q_MIX_path))

    def load_single_model(self,start_string,model_dir="./model"):
        model_files = [f for f in os.listdir(model_dir) if f.startswith(start_string) and f.endswith(".pth")]

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
            return best_model_path
        else:
            print("No valid model file found.")