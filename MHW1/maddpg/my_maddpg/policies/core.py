import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
from torch.nn.functional import softplus
from torch.distributions import Normal
kl_div = torch.distributions.kl_divergence


def mlp_block(input_dim: int,
              output_dim: int,
              normalize=None,
              activation=None,
              initialize=None,
              device=None):
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
                 normalize=None,
                 initialize=None,
                 activation=None,
                 device=None):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (input_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(
                input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(
            mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """
        Returns the output of the Q network.
        Parameters:
            x (Tensor): The input tensor.
        """
        return self.model(x)


class ActorNet_DDPG(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes,
                 normalize=None,
                 initialize=None,
                 activation=None,
                 activation_action=None,
                 device=None):
        super(ActorNet_DDPG, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(
                input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(
            input_shape[0], action_dim, None, activation_action, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor, avail_actions=None):
        logits = self.model(x)
        if avail_actions is not None:
            logits[avail_actions == 0] = -1e10
        return logits
