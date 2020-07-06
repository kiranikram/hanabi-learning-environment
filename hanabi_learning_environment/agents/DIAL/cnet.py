from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

import matplotlib.pyplot as plt

import gin.torch
import numpy as np


def weight_reset(m):
    if isinstance(m, nn.Batchnorm1d) or isinstance(m , nn.Linear):
        m.reset_parameters()
"""
Inputs: tuple of (o_t^a, m_t-1^a', u_t-1^a, a ; 
a: player who's turn it is
 """
 # TODO: add gin configurable ; replace eg num_players with configuarble version
 # TODO: current testing is with comms; option a and b; to be replaced with Batesian belief update
class DRQNet(nn.Module):

    def __init__(self, num_players, rnn_size, comm_size, num_actions , observation_size , init_param_range ):

        super(DRQNet, self).__init__()
        self.num_players = 3
        self.rnn_size = 128
        self.comm_size = 2
        self.num_actions = num_actions
        self.observation_size = observation_size
        self.init_param_range = (-0.08, 0.08)

        #Embedding matrix for DRQN
        self.action_matrix = nn.Embedding(self.num_players,self.rnn_size)
        self.observation_matrix = nn.Embedding(self.observation_size,self.rnn_size)
        self.previous_action_matrix = nn.Embedding(self.num_actions,self.rnn_size)

        #Single layer NN for producing embeddings for messages
        self.message = nn.Sequential(
            nn.BatchNorm1d(self.comm_size),
            nn.Linear(self.comm_size, self.rnn_size),
            nn.ReLu(inplace=True)
        )

        #RNN component for history over POMDP
        self.rnn = nn.GRU(input_size=self.rnn_size, hidden_size=self.rnn_size,num_layers=2,batch_first=True)

        #Output from RNN layer
        self.output = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.BatchNorm1d(self.rnn_size),
            nn.ReLu(),
            nn.Linear(self.rnn_size,self.observation_size)
        )

    def retrieve_parameters(self):
        return list(self.parameters())

    def reset_parameters(self):

        self.rnn.reset_parameters()
        self.action_matrix.reset_parameters()
        self.observation_matrix.reset_parameters()
        self.previous_action_matrix.reset_parameters()
        self.message.apply(weight_reset)
        self.output.appky(weight_reset)
        for x in self.rnn.parameters():
            x.data.uniform_(*self.init_param_range)

    # TODO check reference for observation vs state
    # TODO assert via HLE agent rotations (ie cannot be random) how is offset determined
    def forward(self, observation,messages, hidden_state, previous_action ,agent):

        observation = Variable(torch.LongTensor(observation))
        hidden_state = Variable(torch.LongTensor(hidden_state))
        previous_action = Variable(torch.LongTensor(previous_action))
        agent = Variable(torch.LongTensor(agent))

        Z_A = self.action_matrix(agent)
        Z_O = self.state_matrix(observation)
        Z_U = self.observation_matrix(previous_action)
        Z_M = self.message(messages.view(-1,self.comm_size))

        #Element wise summation of embeddings

        Z = Z_A + Z_O + Z_U + Z_M
        Z = Z.unsqueeze(1)

        rnn_out , H = self.rnn(Z, hidden_state)

        #Retrieve final QNet output Q values from GRU
        OUT = self.output(rnn_out[:,-1,:].squeeze())

        return H, OUT
























