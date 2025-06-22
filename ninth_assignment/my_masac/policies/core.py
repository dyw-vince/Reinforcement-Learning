import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
from torch.nn.functional import softplus
from torch.distributions import Normal
# kl_div = torch.distributions.kl_divergence

# class DiagGaussianDistribution:
#     def __init__(self, action_dim: int):
#         super(DiagGaussianDistribution, self).__init__()
        

class ActivatedDiagGaussianDistribution():
    def __init__(self, action_dim: int, activation_action, device):
        super(ActivatedDiagGaussianDistribution, self).__init__()
        self.mu, self.std = None, None
        self.action_dim = action_dim
        self.distribution = None
        self.activation_fn = activation_action()
        self.device = device

    def set_param(self, mu, std):
        self.mu = mu
        self.std = std
        self.distribution = Normal(mu, std)

    def get_param(self):
        return self.mu, self.std

    def log_prob(self, x):
        return self.distribution.log_prob(x).sum(-1)

    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def stochastic_sample(self):
        return self.distribution.sample()

    def rsample(self):
        return self.distribution.rsample()

    def deterministic_sample(self):
        return self.mu

    def kl_divergence(self, other):
        return torch.distributions.kl_divergence(self.distribution, other.distribution)
    def activated_rsample(self):
        return self.activation_fn(self.rsample())

    def activated_rsample_and_logprob(self):
        act_pre_activated = self.rsample()  # sample without being activated.
        act_activated = self.activation_fn(act_pre_activated)
        log_prob = self.distribution.log_prob(act_pre_activated)
        correction = - 2. * (torch.log(Tensor([2.0])).to(self.device) - act_pre_activated - softplus(-2. * act_pre_activated))
        log_prob += correction
        return act_activated, log_prob.sum(-1)


def mlp_block(input_dim: int,
              output_dim: int,
              normalize = None,
              activation = None,
              initialize = None,
              device = None) :
    block = []
    linear = nn.Linear(input_dim, output_dim, device=device)
    if initialize is not None:
        initialize(linear.weight)
        nn.init.constant_(linear.bias, 0)
    block.append(linear)
    if activation is not None:
        block.append(activation())
    if normalize is not None:
        block.append(normalize(output_dim, device=device))
    return block, (output_dim,)

class CriticNet(Module):
    def __init__(self,
                 input_dim,
                 hidden_sizes,
                 normalize= None,
                 initialize = None,
                 activation = None,
                 device = None):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (input_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """
        Returns the output of the Q network.
        Parameters:
            x (Tensor): The input tensor.
        """
        return self.model(x)


class ActorNet_SAC(Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes,
                 normalize,
                 initialize,
                 activation,
                 activation_action,
                 device= None):
        super(ActorNet_SAC, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        self.output = nn.Sequential(*layers)
        self.out_mu = nn.Linear(hidden_sizes[-1], action_dim, device=device)
        self.out_log_std = nn.Linear(hidden_sizes[-1], action_dim, device=device)
        self.dist = ActivatedDiagGaussianDistribution(action_dim, activation_action, device)

    def forward(self, x: Tensor):
        """
        Returns the stochastic distribution over the continuous action space.
        Parameters:
            x (Tensor): The input tensor.

        Returns:
            self.dist: A distribution over the continuous action space.
        """
        output = self.output(x)
        mu = self.out_mu(output)
        log_std = torch.clamp(self.out_log_std(output), -20, 2)
        std = log_std.exp()
        self.dist.set_param(mu, std)
        return self.dist

