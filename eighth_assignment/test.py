import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from train import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter



class Testing:

    def __init__(self,config,test):
        self.config=config
        self.result_dir = config['result_path']
        self.gif_dir = os.path.join(self.result_dir, 'gif')
        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir)
        self.gif_num = len([file for file in os.listdir(self.gif_dir)])  # current number of gif
        self.runner = Training(config,test)
        self.runner.agent_n.load_model()
        self.agent_names=[agent for agent in self.runner.env.agents]
        # reward of each episode of each agent
        self.episode_rewards = {agent: np.zeros(config['max_train_steps']) for agent in self.runner.env.agents}
        self.epsilon=0
    def run(self):
        for episode in range(self.config['testing_episode']):
            self.runner.agent_n.current_Q_RNN.init_hidden(self.config['N']) #将RNN的隐变量初始化
            states,_= self.runner.env.reset()
            agent_reward = {agent: 0 for agent in self.runner.env.agents}  # agent reward of the current episode
            frame_list = []  # used to save gif
            states=dict2np(states)
            onehot_actions = np.zeros((self.config['N'], self.config['action_dim']))
            for episode_step in range(self.runner.config['max_episode_times']):
                actions,actions_dict,onehot_actions = self.runner.agent_n.choose_action(self.agent_names,states,onehot_actions,self.epsilon) 
                # need to transit 'a_n' into dict
                next_states_n, rewards_n, dones, _, _ = self.runner.env.step(actions_dict)
                frame_list.append(Image.fromarray(self.runner.env.render()))  # 第二次frame_list=[]时报错
                states = dict2np(next_states_n)
                for agent_id, reward in rewards_n.items():  # update reward
                    agent_reward[agent_id] += reward
            # env.close()
            message = f'episode {episode + 1}, '
            # episode finishes, record reward
            for agent_id, reward in agent_reward.items():
                self.episode_rewards[agent_id][episode] = reward
                message += f'{agent_id}: {reward:>4f}; '
            print(message)
            # save gif
            frame_list[0].save(os.path.join(self.gif_dir, f'out{self.gif_num + episode + 1}.gif'),
                            save_all=True, append_images=frame_list[1:], duration=1, loop=0)
            self.draw(self.config)
    
    def draw(self,config):
        # 解析命令行参数
        rewards = np.load(config['reward_path'])
        print(len(rewards))
        smooth_rewards = savgol_filter(rewards, window_length=11, polyorder=3)
        fig, ax = plt.subplots()
        x = range(1, 5*np.shape(rewards)[0]+1,5)
        ax.plot(x, rewards, label="agent0_1_2", alpha=0.3)
        ax.plot(x, smooth_rewards, label="Smoothed", color='red')
        ax.legend()
        ax.set_xlabel('episode/K')
        ax.set_ylabel('reward')
        title = f'evaluate result of qmix'
        ax.set_title(title)
        plt.savefig('rewards_over_episodes.png')
        plt.show()