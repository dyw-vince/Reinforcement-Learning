import os
import torch
from torch import nn
from torch import Tensor
from operator import itemgetter
MAX_GPUs = 100


class MADDPG_learner():
    def __init__(self,
                 config,
                 model_keys,
                 agent_keys,
                 policy):
        super(MADDPG_learner, self).__init__()
        self.value_normalizer = None
        self.config = config
        self.n_agents = config.n_agents
        self.dim_id = self.n_agents

        self.model_keys = model_keys
        self.agent_keys = agent_keys
        self.episode_length = config.episode_length
        self.learning_rate = config.learning_rate if hasattr(
            config, 'learning_rate') else None
        self.use_linear_lr_decay = config.use_linear_lr_decay if hasattr(
            config, 'use_linear_lr_decay') else False
        self.end_factor_lr_decay = config.end_factor_lr_decay if hasattr(
            config, 'end_factor_lr_decay') else 0.5
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        self.use_actions_mask = config.use_actions_mask if hasattr(
            config, 'use_actions_mask') else False
        self.policy = policy
        self.optimizer = None
        self.scheduler = None
        self.use_grad_clip = config.use_grad_clip
        self.grad_clip_norm = config.grad_clip_norm
        self.device = config.device
        self.model_dir = config.model_dir
        self.running_steps = config.running_steps
        self.use_parameter_sharing = config.use_parameter_sharing
        self.iterations = 0

        self.optimizer = {
            key: {'actor': torch.optim.Adam(self.policy.parameters_actor[key], self.config.learning_rate_actor, eps=1e-5),
                  'critic': torch.optim.Adam(self.policy.parameters_critic[key], self.config.learning_rate_critic, eps=1e-5)}
            for key in self.model_keys}
        self.scheduler = {
            key: {'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer[key]['actor'],
                                                             start_factor=1.0,
                                                             end_factor=self.end_factor_lr_decay,
                                                             total_iters=self.config.running_steps),
                  'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer[key]['critic'],
                                                              start_factor=1.0,
                                                              end_factor=self.end_factor_lr_decay,
                                                              total_iters=self.config.running_steps)}
            for key in self.model_keys}
        self.gamma = config.gamma
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=False)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        IDs = sample_Tensor['agent_ids']
        bs = batch_size
        obs_joint = self.get_joint_input(obs, (batch_size, -1))
        next_obs_joint = self.get_joint_input(obs_next, (batch_size, -1))
        actions_joint = self.get_joint_input(actions, (batch_size, -1))

        # get actions
        _, actions_eval = self.policy(observation=obs, agent_ids=IDs)
        _, actions_next = self.policy.Atarget(
            next_observation=obs_next, agent_ids=IDs)
        # get values
        actions_next_joint = self.get_joint_input(
            actions_next, (batch_size, -1))
        _, q_eval = self.policy.Qpolicy(
            joint_observation=obs_joint, joint_actions=actions_joint, agent_ids=IDs)
        _, q_next = self.policy.Qtarget(joint_observation=next_obs_joint, joint_actions=actions_next_joint,
                                        agent_ids=IDs)
        for key in self.model_keys:
            mask_values = agent_mask[key]
            # update critic
            q_eval_a = q_eval[key].reshape(bs)
            q_next_i = q_next[key].reshape(bs)
            q_target = rewards[key] + \
                (1 - terminals[key]) * self.gamma * q_next_i
            td_error = (q_eval_a - q_target.detach()) * mask_values
            loss_c = (td_error ** 2).sum() / mask_values.sum()
            self.optimizer[key]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters_critic[key], self.grad_clip_norm)
            self.optimizer[key]['critic'].step()
            if self.scheduler[key]['critic'] is not None:
                self.scheduler[key]['critic'].step()

            # update actor
            a_joint = {
                k: actions_eval[k] if k == key else actions[k] for k in self.agent_keys}
            act_eval = self.get_joint_input(a_joint, (batch_size, -1))
            _, q_policy = self.policy.Qpolicy(joint_observation=obs_joint, joint_actions=act_eval,
                                              agent_ids=IDs, agent_key=key)
            q_policy_i = q_policy[key].reshape(bs)
            loss_a = -(q_policy_i * mask_values).sum() / mask_values.sum()
            self.optimizer[key]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters_actor[key], self.grad_clip_norm)
            self.optimizer[key]['actor'].step()
            if self.scheduler[key]['actor'] is not None:
                self.scheduler[key]['actor'].step()

            learning_rate_actor = self.optimizer[key]['actor'].state_dict()[
                'param_groups'][0]['lr']
            learning_rate_critic = self.optimizer[key]['critic'].state_dict()[
                'param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate_actor": learning_rate_actor,
                f"{key}/learning_rate_critic": learning_rate_critic,
                f"{key}/loss_actor": loss_a.item(),
                f"{key}/loss_critic": loss_c.item(),
                f"{key}/predictQ": q_eval[key].mean().item()
            })

        self.policy.soft_update(self.tau)
        return info

    def build_training_data(self, sample,
                            use_parameter_sharing=False,
                            use_actions_mask=False,
                            use_global_state=False):

        batch_size = sample['batch_size']
        seq_length = 1
        state, avail_actions, filled = None, None, None
        obs_next, state_next, avail_actions_next = None, None, None
        IDs = None
        obs = {k: Tensor(sample['obs'][k]).to(self.device)
               for k in self.agent_keys}
        actions = {k: Tensor(sample['actions'][k]).to(
            self.device) for k in self.agent_keys}
        rewards = {k: Tensor(sample['rewards'][k]).to(
            self.device) for k in self.agent_keys}
        terminals = {k: Tensor(sample['terminals'][k]).float().to(
            self.device) for k in self.agent_keys}
        agent_mask = {k: Tensor(sample['agent_mask'][k]).float().to(
            self.device) for k in self.agent_keys}
        obs_next = {k: Tensor(sample['obs_next'][k]).to(
            self.device) for k in self.agent_keys}
        if use_actions_mask:
            avail_actions = {k: Tensor(sample['avail_actions'][k]).float().to(
                self.device) for k in self.agent_keys}
            avail_actions_next = {k: Tensor(sample['avail_actions_next'][k]).float().to(
                self.device) for k in self.model_keys}

        if use_global_state:
            state = Tensor(sample['state']).to(self.device)
            state_next = Tensor(sample['state_next']).to(self.device)

        sample_Tensor = {
            'batch_size': batch_size,
            'state': state,
            'state_next': state_next,
            'obs': obs,
            'actions': actions,
            'obs_next': obs_next,
            'rewards': rewards,
            'terminals': terminals,
            'agent_mask': agent_mask,
            'avail_actions': avail_actions,
            'avail_actions_next': avail_actions_next,
            'agent_ids': IDs,
            'filled': filled,
            'seq_length': seq_length,
        }
        return sample_Tensor

    def get_joint_input(self, input_tensor, output_shape=None):
        if self.n_agents == 1:
            joint_tensor = itemgetter(*self.agent_keys)(input_tensor)
        else:
            joint_tensor = torch.concat(itemgetter(
                *self.agent_keys)(input_tensor), dim=-1)
        if output_shape is not None:
            joint_tensor = joint_tensor.reshape(output_shape)
        return joint_tensor

    def save_model(self, model_path):
        torch.save(self.policy.state_dict(), model_path)

    def load_model(self, path, model=None):
        file_names = os.listdir(path)
        if model is not None:
            path = os.path.join(path, model)
            if model not in file_names:
                raise RuntimeError(
                    f"The folder '{path}' does not exist, please specify a correct path to load model.")
        else:
            for f in file_names:
                if "seed_" not in f:
                    file_names.remove(f)
            file_names.sort()
            path = os.path.join(path, file_names[-1])

        model_names = os.listdir(path)
        if os.path.exists(path + "/obs_rms.npy"):
            model_names.remove("obs_rms.npy")
        if len(model_names) == 0:
            raise RuntimeError(f"There is no model file in '{path}'!")
        model_names.sort()
        model_path = os.path.join(path, model_names[-1])
        self.policy.load_state_dict(torch.load(str(model_path), map_location={
            f"cuda:{i}": self.device for i in range(MAX_GPUs)}))
        print(f"Successfully load model from '{path}'.")
