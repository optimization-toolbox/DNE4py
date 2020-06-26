import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
torch.set_default_dtype(torch.float64)


class MLP(nn.Module):
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
            l = nn.Linear(self.layers[i - 1], self.layers[i])
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

        if isinstance(m, nn.Linear):

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
