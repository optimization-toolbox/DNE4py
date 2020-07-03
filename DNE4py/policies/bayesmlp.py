import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
torch.set_default_dtype(torch.float64)


class BayesMLP(nn.Module):
    r"""Class for Deep Neural Network as Policy Model"""

    def __init__(self, policy_config):

        super().__init__()
        # Parameters:
        self.layers = policy_config['layers']
        self.activations = policy_config['activations']

        self.seed = policy_config['seed']
        self.initialization_method = policy_config['initialization_method']
        self.initialization_param = policy_config['initialization_param']

        sequential_layers = []
        for i in range(1, len(self.layers)):
            l = BayesLinear(self.layers[i - 1], self.layers[i])
            if self.activations[i - 1] == 'tanh':
                l2 = nn.Tanh()
            sequential_layers.append(l)
            sequential_layers.append(l2)

        self.model = nn.Sequential(*sequential_layers)

        # Init weights:
        torch.manual_seed(self.seed)
        self.model.apply(self.weight_init)
        torch.seed()

    def weight_init(self, m):
        if isinstance(m, BayesLinear):
            for n, param in m.named_parameters():
                if 'bias' in n:
                    init.zeros_(param.data)
                elif 'weight' in n:
                    if self.initialization_method == 'uniform':
                        init.uniform_(param.data,
                                      -self.initialization_param,
                                      self.initialization_param)
                    elif self.initialization_method == 'normal':
                        init.normal_(param.data, mean=0.0, std=self.initialization_param)
                    elif self.initialization_method == 'xavier':
                        gain = init.calculate_gain('tanh')
                        init.xavier_normal_(param.data, gain)

    def forward(self, observation):

        with torch.no_grad():

            # Input:
            observation = torch.from_numpy(observation)

            # Sampler:
            prediction = self.model(observation)
            return prediction

    def get_weights(self):
        parameters = np.concatenate([p.detach().numpy().ravel() for p in self.model.parameters()])
        return parameters

    def set_weights(self, new_weights):
        last_slice = 0
        for p in self.model.parameters():
            size_layer_parameters = np.prod(np.array(p.data.size()))
            new_parameters = new_weights[last_slice:last_slice + size_layer_parameters].reshape(p.data.shape)
            last_slice += size_layer_parameters
            p.data = torch.from_numpy(new_parameters).detach()


import math

import torch
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F


class BayesLinear(Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))

        self.bias = bias
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
        else:
            print("Bias should be True")
            exit()

    def forward(self, input):

        weight = self.weight_mu

        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)

        return F.linear(input, weight, bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)
