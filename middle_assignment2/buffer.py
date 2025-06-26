import numpy as np
class RND_Buffer:
    def __init__(self):
        self.create_memory()

    def create_memory(self):
        self.states=[]
        self.next_states=[]
        self.actions=[]
        self.rewards=[]
        self.int_rewards=[]
        self.dones=[]
        self.ext_values=[]
        self.int_values=[]
        self.actor_dists=[]
        self.actor_dists_np=[]

    def store(self, state, next_state, action, reward, int_reward, done, ext_value, int_value, actor_dist, actor_dist_np):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.int_rewards.append(int_reward)
        self.dones.append(done)
        self.ext_values.append(ext_value)
        self.int_values.append(int_value)
        self.actor_dists.append(actor_dist)
        self.actor_dists_np.append(actor_dist_np)

    def store_last(self, ext_value, int_value):
        self.ext_values.append(ext_value)
        self.int_values.append(int_value)
    
    def get_data(self):
        states =np.stack(self.states).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        next_states = np.stack(self.next_states).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        actions = np.stack(self.actions).transpose().reshape([-1])
        rewards = np.stack(self.rewards).transpose().clip(-1, 1)
        int_rewards = np.stack(self.rewards).transpose()
        dones = np.stack(self.dones).transpose()
        ext_values = np.stack(self.ext_values).transpose()
        int_values = np.stack(self.int_values).transpose()
        actor_dists = self.actor_dists
        actor_dists_np = np.vstack(self.actor_dists_np)
        return states, next_states, actions, rewards, int_rewards, dones, ext_values, int_values, actor_dists, actor_dists_np
    
    def clear(self):
        self.create_memory()
