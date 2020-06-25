from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from .torchbnn import *
# import torchbnn as bnn


class Sampler_LSTM(nn.Module):
    r"""Class for Bayesian Deep Neural Network as Policy Model"""

    def __init__(self, policy_config):

        super().__init__()

        # Parameters:
        self.seed = policy_config['seed']
        self.x_dim = policy_config['x_dim']
        self.lambda_ = policy_config['lambda_']
        self.initialization = policy_config['initialization']

        self.input_size = (self.x_dim * (self.lambda_)) + (self.x_dim * (self.lambda_ + 1)) + 1 + 1

        self.model = nn.ModuleDict(OrderedDict({

            'lstm':
            nn.LSTM(
                self.input_size,
                256,
                5,
                batch_first=True
            ),

            'bdnn':
            nn.Sequential(
                BayesLinear2(0.0, 1.0, 256, self.x_dim),
                nn.Sigmoid()
            )

        }))

        # Init weights:
        torch.manual_seed(self.seed)
        self.model.apply(self.weight_init)
        torch.manual_seed(4040)

        # Init Hidden and cell state:
        self.reset()

    def weight_init(self, m):

        if isinstance(m, nn.LSTM) or isinstance(m, BayesLinear2):

            for n, param in m.named_parameters():
                if 'bias' in n:
                    init.zeros_(param.data)
                elif 'weight' in n:

                    #init.uniform_(param.data, -self.initialization, self.initialization)
                    init.normal_(param.data, mean=0.0, std=self.initialization)

    def forward(self, observation, prev_r, prev_prediction, t, return_dist=False):

        with torch.no_grad():

            # Input:
            observation = torch.from_numpy(observation).float().flatten()
            prev_r = torch.from_numpy(np.array([prev_r])).float()
            prev_prediction = torch.from_numpy(prev_prediction).float().flatten()
            t = torch.from_numpy(np.array([t])).float()

            #print("\n ---------- Forward ----------")
            # print(observation)
            # print(prev_r)
            # print(prev_prediction)
            # print(t)
            #print(" ---------- End Forward ----------")

            input_vec = torch.cat((observation, prev_r, prev_prediction, t))
            # LSTM:
            input_vec = input_vec.view(-1, 1, self.input_size)
            lstm_enc, (self.lstm_hidden, self.lstm_cell) = self.model['lstm'](input_vec, (self.lstm_hidden, self.lstm_cell))
            lstm_enc = lstm_enc[-1, -1]

            # Sampler:
            prediction = torch.Tensor(self.lambda_, self.x_dim)
            for i in range(len(prediction)):
                prediction[i] = self.model['bdnn'](lstm_enc)
            prediction = prediction.numpy()

            # Output:
            if return_dist == False:
                return prediction, None

            elif return_dist == True:

                dist = torch.Tensor(1000, self.x_dim)
                for i in range(len(dist)):
                    dist[i] = self.model['bdnn'](lstm_enc)
                dist = dist.numpy()
                return prediction, dist

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
            # p.data = new_parameters.clone().detach()

    def reset(self):

        # Reset hidden and cell states of offspring_generator
        self.lstm_hidden = torch.zeros(5,
                                       1,
                                       256)
        self.lstm_cell = torch.zeros(5,
                                     1,
                                     256)
