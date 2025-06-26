from learner import *
from envs import *
from utils import *
from buffer import *
from torch.multiprocessing import Pipe
import wandb
import socket
import pickle
import numpy as np


# def main():
class RNDAgent:
    def __init__(self):
        super(RNDAgent, self).__init__()
        config=get_config(config_path="config.yaml")
        self.config=config
        self.device=config.Device
    #------------init env------------------
        self.env_id=config.EnvID
        print(self.env_id)
        self.env_type=config.EnvType
        env = gym.make(self.env_id)
        self.input_size = env.observation_space.shape  # 4
        self.output_size = env.action_space.n  # 2
        env.close()
        self.num_worker = config.NumEnv
        self.num_frame = config.StateStackSize
        self.pic_height=config.PreProcHeight
        self.pic_width=config.ProProcWidth
        self.works = []
        self.parent_conns = []
        self.child_conns = []
        

    #----------init training config-----------
        self.num_step = config.NumStep #一次交互的最大步数
        self.epoch = config.Epoch #训练的epoch数
        self.mini_batch = config.MiniBatch #4
        self.batch_size = int(self.num_step * self.num_worker / self.mini_batch) #2048
        self.learning_rate = config.LearningRate
        self.max_step = config.MaxStep
        self.sample_episode = 0 
        self.sample_all_reward = 0
        self.sample_step = 0
        self.sample_env_idx = 0
        self.sample_i_all_reward = 0
        self.global_update = 0
        self.global_step = 0

    #------------init net config--------------
        self.lam = config.Lambda #GAE中的衰减因子
        self.ppo_eps = config.PPOEps #防止除数为0
        self.entropy_coef = config.Entropy #熵的系数
        self.gamma = config.Gamma  #外在奖励的折扣因子
        self.int_gamma = config.IntGamma #内在奖励的折扣因子
        self.clip_grad_norm = config.ClipGradNorm
        self.clip_range = config.ClipRange
        self.ext_coef = config.ExtCoef #外在奖励系数
        self.int_coef = config.IntCoef #内在奖励系数
        self.update_proportion=config.UpdateProportion
        self.sticky_action = config.StickyAction 
        self.action_prob = config.ActionProb
        self.life_done = config.LifeDone
        self.pre_obs_norm_step = config.ObsNormStep #开始训练前随机采样obs_norm的最大步数

        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(1, 1, self.pic_height, self.pic_width))
        self.discounted_reward = RewardForwardFilter(self.int_gamma)

    #--------------init path-------------------
        self.model_path = config.ModelPath
        self.predictor_path = config.PredictorPath
        self.target_path = config.TargetPath
        self.log_path = config.LogPath


    #--------------choose options---------------
        self.is_load_model = config.IsLoadModel
        self.is_render = config.IsRender
        

    #---------------init wandb-----------------
        config_dict = vars(config)
        time_string = get_time_string()
        create_directory(str(self.log_path))
        wandb.init(config=config_dict,
                    project=config.ProjectName,
                    entity=config.WandbUserName,
                    notes=socket.gethostname(),
                    dir=self.log_path,
                    group=config.EnvID,
                    job_type=config.Agent,
                    name=time_string,
                    reinit=True,
                    settings=wandb.Settings(start_method="fork")
                    )
        
        for idx in range(self.num_worker):
            parent_conn, child_conn = Pipe()
            work = AtariEnvironment(self.env_id, self.is_render, idx, child_conn, 
            sticky_action=self.sticky_action, p=self.action_prob,life_done=self.life_done)
            work.start()
            self.works.append(work)
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

        self.build_learner()
        self.build_buffer()
        
    def build_buffer(self):
        self.buffer=RND_Buffer()

    def build_learner(self):
        self.learner = RNDLearner(
            input_size = self.input_size, output_size = self.output_size,
            num_env = self.num_worker,  num_step = self.num_step,
            gamma = self.gamma, lam = self.lam,
            learning_rate = self.learning_rate,
            ent_coef = self.entropy_coef,
            clip_grad_norm = self.clip_grad_norm,
            epoch = self.epoch,
            batch_size = self.batch_size,
            ppo_eps = self.ppo_eps,
            update_proportion= self.update_proportion,
            device = self.device,
            config= self.config
        )
        if self.is_load_model:
            print('load model...')
            self.learner.model.load_state_dict(torch.load(self.model_path))
            self.learner.rnd.predictor.load_state_dict(torch.load(self.predictor_path))
            self.learner.rnd.target.load_state_dict(torch.load(self.target_path))
            print('load finished!')
            
    def init_obs_rms(self):
        # normalize obs
        next_obs = []
        for step in range(self.num_step * self.pre_obs_norm_step):
            actions = np.random.randint(0, self.output_size, size=(self.num_worker,))

            for parent_conn, action in zip(self.parent_conns, actions):
                parent_conn.send(action)

            for parent_conn in self.parent_conns:
                s, r, d, rd, lr = parent_conn.recv()
                next_obs.append(s[3, :, :].reshape([1, 84, 84]))

            if len(next_obs) % (self.num_step * self.num_worker) == 0:
                next_obs = np.stack(next_obs)
                self.obs_rms.update(next_obs)
                next_obs = []
        print('End to initalize...')
    
    def training(self):
        states = np.zeros([self.num_worker, self.num_frame, self.pic_height, self.pic_width])
        self.init_obs_rms()
        while True:
            self.global_step += (self.num_worker * self.num_step)
            self.global_update += 1

            # Step 1. n-step rollout
            for _ in range(self.num_step):
                actions, value_ext, value_int, actor_dist = self.learner.get_action(states)

                for parent_conn, action in zip(self.parent_conns, actions):
                    parent_conn.send(action)

                next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                for parent_conn in self.parent_conns:
                    s, r, d, rd, lr = parent_conn.recv()
                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d)
                    real_dones.append(rd)
                    log_rewards.append(lr)
                    next_obs.append(s[3, :, :].reshape([1, self.pic_width, self.pic_height]))

                next_states = np.stack(next_states)
                rewards = np.hstack(rewards)
                dones = np.hstack(dones)
                real_dones = np.hstack(real_dones)
                next_obs = np.stack(next_obs)

                # total reward = int reward + ext Reward
                int_rewards = self.learner.compute_int_reward(
                    self._process_obs(next_obs))
                int_rewards = np.hstack(int_rewards)
                self.sample_i_all_reward += int_rewards[self.sample_env_idx]
                
                self.buffer.store(states,next_obs,actions,rewards,int_rewards,dones,value_ext,value_int,actor_dist,actor_dist.cpu().numpy())
                states = next_states[:, :, :, :]
                self.sample_all_reward += log_rewards[self.sample_env_idx]

                self.sample_step += 1

                if real_dones[self.sample_env_idx]:
                    self.sample_episode += 1
                    wandb.log({
                        'reward_per_epi': self.sample_all_reward,
                        'reward_per_rollout': self.sample_all_reward,
                        'step': self.sample_step,
                    }, step=self.global_update)
                    self.sample_all_reward = 0
                    self.sample_step = 0
                    self.sample_i_all_reward = 0

            # calculate last next value
            _, value_ext, value_int, _ = self.learner.get_action(states)
            self.buffer.store_last(value_ext,value_int)
            
            #-------------------------------------------------------------------------------------------

            total_states, total_next_obses, total_actions, total_rewards, total_int_rewards, total_dones, total_ext_values, total_int_values, total_actor_dists, total_actor_dists_np= self.buffer.get_data()

            total_reward_per_env = np.array([self.discounted_reward.update(reward_per_step) for reward_per_step in
                                            total_int_rewards.T])
            self.reward_rms.update(total_reward_per_env)


            total_int_rewards /= np.sqrt(self.reward_rms.var)
            wandb.log({
                    "data/int_reward_per_epi": np.sum(total_int_rewards) / self.num_worker,
                    "data/int_reward_per_rollout": np.sum(total_int_rewards) / self.num_worker,
                    "data/max_prob": softmax(total_actor_dists_np).max(1).mean()
                }, step=self.sample_episode)
           
            ext_target, ext_adv = self.make_train_data(total_rewards, total_dones, total_ext_values)
            
            int_target, int_adv = self.make_train_data(total_int_rewards, 
                                        np.zeros_like(total_int_rewards), total_int_values)
            
            total_adv = int_adv * self.int_coef + ext_adv * self.ext_coef

            self.obs_rms.update(total_next_obses)
            # Step 5. Training!
            self.learner.train_model(np.float32(total_states) / 255., ext_target, int_target, total_actions,
                            total_adv, self._process_obs(total_next_obses),
                            total_actor_dists)
            self.buffer.clear()

            if self.global_step % (self.num_worker * self.num_step * 100) == 0:
                print('Now Global Step :{}'.format(self.global_step))
                torch.save(self.learner.model.state_dict(), self.model_path)
                torch.save(self.learner.rnd.predictor.state_dict(), self.predictor_path)
                torch.save(self.learner.rnd.target.state_dict(), self.target_path)

        wandb.finish()

    def testing(self):
        states = np.zeros([self.num_worker, self.num_frame, self.pic_height, self.pic_width])
        steps = 0
        rall = 0
        rd = False
        int_reward_list = []
        while not rd:
            steps += 1
            actions, value_ext, value_int, policy = self.learner.get_action(states)

            for parent_conn, action in zip(self.parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            for parent_conn in self.parent_conns:
                s, r, d, rd, lr = parent_conn.recv()
                rall += r
                next_states = s.reshape([1, self.num_frame, self.pic_height, self.pic_width])
                next_obs = s[3, :, :].reshape([1, 1, self.pic_height, self.pic_width])

            # total reward = int reward + ext Reward
            int_reward = self.learner.compute_int_reward(next_obs)
            int_reward_list.append(int_reward)
            states = next_states[:, :, :, :]
            
            if rd:
                intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(
                    intrinsic_reward_list)
                with open('int_reward', 'wb') as f:
                    pickle.dump(intrinsic_reward_list, f)
                steps = 0
                rall = 0
         
    def _process_obs(self,obs):
        return ((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-self.clip_range, self.clip_range)

    def make_train_data(self,reward, done, value):
        discounted_return = np.empty([self.num_worker, self.num_step])
        gae = np.zeros_like([self.num_worker, ])
        for t in range(self.num_step - 1, -1, -1):
            delta = reward[:, t] + self.gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
            gae = delta + self.gamma * self.lam * (1 - done[:, t]) * gae

            discounted_return[:, t] = gae + value[:, t]
        adv = discounted_return - value[:, :-1]
        return discounted_return.reshape([-1]), adv.reshape([-1])
    
if __name__ == '__main__':
    agent=RNDAgent()
    agent.training()
