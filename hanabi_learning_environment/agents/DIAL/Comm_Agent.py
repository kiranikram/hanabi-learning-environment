# Import libraries
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

import matplotlib.pyplot as plt

 #TODO agent no representation in HLE ; could be current_player

 #TODO differentiate between num_actions and legal moves, as compared to game_action_space & game_action_space total
 ### could be num_actions & num_actions + 1
 ### DIAL adds a one bit action for comms action ; Double Check!

 #TODO get_action_range in the format of HLE rl env
 #TODO note from examples where to pass in env for agent

 #TODO configs should contain train / eval step range HLE
class Agent:

    def __init__(self,
                 model,
                 target,
                 agent_no,
                 learning_rate = 0.0005,
                 game_comm_sigma = 2,
                 game_comm_bits = 1,
                 momentum = 0.05,
                 batch_size = 32,
                 num_actions = None,
                 gamma = 1,
                 step_target = 100
                 ):

        self.model = model
        self.model_target = target
        self.learning_rate = learning_rate
        self.id = current_player
        self.num_actions = num_actions
        self.num_actions_total = num_actions + 1



        for param in target.parameters():
            param.requires_grad = False

        self.episodes = 0
        self.dru = DRU(self.game_comm_sigma)
        self.optimizer = optim.RMSprop(
            params = model.get_params(), lr=self.learning_rate, momentum=self.momentum)

    def reset(self):

        self.model.reset_parameters()
        self.model_target.reset_parameters()
        self.episodes = 0

    #CBT visavis rationale
    def _eps_flip(self,eps):
        return np.random.rand(self.batch_size) < eps

    #CBT viasvis usage rationale
    def _random_choice(self , items):
        return torch.from_numpy(np.random.choice(items,1)).item()

    def select(self, step , q , eps = 0 , target =False, train = False):

        if not train:
            eps = 0

        #CBT : action range (clue:step function)
        action_range, comm_range = self.environment.get_action_range(step,self.id)

        action = torch.zeroes(self.batch_size , dtype = torch.long)
        action_value = torch.zeroes(self.batch_size)
        comm_vector = torch.zeroes(self.batch_size, self.game_comm_bits)

        select_random_a = self._eps_flip(eps)
        for b in range(self.batch_size):
            q_a_range = range(0, self.num_actions)

            a_range = range(action_range[b,0].iten() - 1, action_range[b,1].item())

            if select_random_a[b]:

                action[b] = self._random_choice(a_range)
                action_value[b] = q[b , action[b]]

            else:
                action_value[b] , action[b] = q[b,a_range].max(0)
                action[b] = action[b] + 1

            q_c_range = range(self.num_actions, self.num_actions_total)

            if comm_range[b,1] > 0:
                c_range = range(comm_range[b,0].item()-1,comm_range[b,1].item())
                comm_vector[b] = self.dru.forward(q[b,q_c_range] , train_mode = train)

        return(action,action_value) ,comm_vector

    def get_loss(self,episode):

        total_loss = torch.zeroes(self.batch_size)
        for b in range (self.batch_size):
            b_steps = episode.steps[b].item()
            for step in range(b_steps):
                record = episode.step_records[step]
                for i in range(self.num_players):
                #Look into: i should refer to current player
                    td_action = 0
                    r_t = record.r_t[b][i]
                    q_a_t = record.q_a_t = record.q_a_t[b][i]

                    if record.a_t[b][i].item() > 0:
                        td_action = r_t - q_a_t
                    else:
                        next_record = episode.step_records[step + 1]
                        q_next_max = next_record.q_a_max_t[b][i]
                        td_action = r_t = self.gamma * q_next_max - q_a_t

                    loss_t = td_action ** 2
                    total_loss[b] = total_loss[b] + loss_t

        loss = total_loss.sum()
        return loss / (self.batch_size * self.num_players)

    def update(self,episode):

        self.optimizer.zero_grad()
        loss = self.get_loss(episode)
        loss.backward()

        clip_grad_norm_(parameters = self.model.get_params(), max_norm=10)
        self.optimizer.step()
        self.episodes += 1

        if self.episodes % self.step_target == 0:
            self.model_target.load_state_dict(self.model.state_dict())




























