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


# TODO look into how to incorporate env dynamics
 # TODO how are agents rotated; refer to agent_idx
 # TODO how are targets for agents updated seperately (ref: [agent_idx,:]

class DotDic(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))


class Arena:

    def __init__(self, environment,
                 game_comm_bits=xx,
                 num_players=None,
                 batch_size=32,
                 rnn_size=128,
                 eps_decay = xx):

        self.eps = 0.05
        self.num_players = num_players
        self.batch_size = batch_size
        self.game_comm_bits = game_comm_bits
        self.rnn_size = rnn_size
        self.eps_decay = eps_decay

    def create_episode(self):
        pass

    def create_step_record(self):
        record = DotDic({})
        record.s_t = None
        record.r_t = torch.zeros(self.batch_size, self.num_players)
        record.terminal = torch.zeros(self.batch_size)

        record.agent_inputs = []
        record.a_t = torch.zeros(self.batch_size, self.num_players, dtype=torch.long)
        record.comm = torch.zeros(self.batch_size, self.num_players, self.game_comm_bits)
        record.comm_target = record.comm.clone()

        record.hidden = torch.zeros(self.num_players, 2, self.batch_size, self.rnn_size)
        record.hidden_target = torch.zeros(self.num_players, 2, self.batch_size, self.rnn_size)

        record.q_a_t = torch.zeros(self.batch_size,self.num_players)
        record.q_a_max_t = torch.zeros(self.batch_size, self.num_players)

        return record

    def run_episode(self,agents,train_mode=False):

        game = self.game.reset()
        self.eps = self.eps * self.eps_decay

        step = 0
        episode = self.create_episode()
        s_t = game.get_state

        episode.step_records.append(self.create_step_record())
        episode.step_records[-1].s_t = s_t

        #will change nsteps to reflect total episode steps HLE
        episode_steps = train_mode and self.nsteps +1 or self.nsteps

        #need to see how HLE treats batch size
        while step < epsisode_steps and episode.ended.sum() <self.batch_size:
            episode.step_records.append(self.create_step_record())

            #how does HLE iterate through agents
            for i in range(1,self.num_players+1):
                agent = agents[i]
                agent_idx = i - 1

                #array/datastructure ; see how comm is updated
                comm = episode.step_records[step].comm.clone()
                comm_limited = self.game.get_comm_limited(step, agent.id)
                if comm_limited is not None:
                    comm_lim = torch.zeros(self.batch_size,1,self.game_comm_bits)
                    for b in range(self.batch_size):
                        if comm_limited[b].item() > 0:
                            comm_lim[b] = comm[b][comm_limited[b] -1]
                    comm = comm_lim
                else:
                    comm[:, agent_idx].zero_()

                previous_action = torch.ones(self.batch_size, dtype=torch.long)

                for b in range(self.batch_size):
                    if step > 0 and episode.step_records[step-1].a_t[b,agent_idx] > 0:
                        previous_action[b] = episode.step_records[step-1].a_t[b,agent_idx]
                batch_agent_index = torch.zeros(self.batch_size, dtype = torch.long).fill_(agent_idx)

                agent_inputs = {
                    'observation' : episode.step_records[step].s_t[:,agent_idx],
                    'messages':comm,
                    'hidden_state':episode.step_records[step].hidden_state[agent_idx,:],
                    'previous_action': previous_action,
                    'agent':batch_agent_index
                }
                episode.step_records[step].agent_inputs.append(agent_inputs)

                #Q Value Retreival from DRQNet
                hidden_t, q_t = agent.model(**agent_inputs)
                episode.step_records[step+1].hidden_state[agent_idx] = hidden_t.squeeze()

                #Picking actions
                (action, action_value), comm_vector = agent.select(step, q_t, eps= self.eps, train=train_mode)

                episode.step_records[step].a_t[:,agent_idx] = action
                episode.step_records[step].q_a_t[:, agent_idx] = action_value
                episode.step_records[step+1].comm[:, agent_idx] = comm_vector


            a_t = episode.step_records[step].a_t
            episode.step_records[step].r_t,episode.step_records[step].terminal = self.environment.step(a_t)

            #need to change NSTEPS to HLE version
            if step < self.nsteps:
                for b in range (self.batch_size):
                    if not episode.ended[b]:
                        episode.steps[b] = episode.steps[b] + 1
                        episode.r[b] = episode.r[b] + episode.step_records[step].r_t[b]
                    if episode.step_records[step].terminal[b]:
                        episode.ended[b] = 1

            #update target network during training
            if train_mode:
                for i in range(1, self.num_players +1):
                    agent_target = agents[i]
                    agent_idx = i- 1

                    agent_inputs = episode.step_records[step].agent_inputs[agent_idx]
                    comm_target = agent_inputs.get('messages' , None)

                    comm_target = episode.step_records[step].comm_target.clone()
                    comm_limited = self.environment.get_comm_limited(step,agent.id)
                    if comm_limited is not None:
                        comm_lim = torch.zeros(self.batch_size, 1, self.game_comm_bits)
                        for b in range(self.batch_size):
                            if comm_limited[b].item() > 0:
                                comm_lim[b] = comm_target[b][comm_limited[b] -1]
                        comm_target = comm_limited
                    else:
                        comm_target[:,agent_idx].zero_()

                    agent_target_inputs = copy.copy(agent_inputs)
                    agent_target_inputs['messages'] = Variable(comm_target)
                    agent_target_inputs['hidden'] = episode.step_records[step].hidden_target[agent_idx, :]
                    hidden_target_t, q_target_t = agent_target(**agent_target_inputs)

                    episode.step_records[step+1].hidden_target[agent_idx]=hidden_target_t.squeeze()

                    (action,action_value),comm_vector = agent_target.select(step,q_target_t, eps=0, target = True, train=True)

                    episode.step_record[step].q_a_max_t[:,agent_idx]= action_value
                    episode.step_record[step + 1].comm_target[:, agent_idx] = comm_vector

            step = step+1
            if episode.ended.sum().item() <self.batch_size:
                episode.step_records[step].s_t = self.environment.get_state()

        episode.game_stats = self.environment.get_stats(episode.steps)

        return episode

    def average_reward(self, episode):

        reward = episode.r.sum() / (self.opt.bs * self.opt.game_nagents)
        if normalized:
            oracle_reward = episode.game_stats.oracle_reward.sum() / self.opt.bs
            if reward == oracle_reward:
                reward = 1
            elif oracle_reward == 0:
                reward = 0
            else:
                reward = reward / oracle_reward
        return float(reward)

    def train(self, agents,reset = True, verbose = False, test_callback = None):

        if reset:
            for agent in agents[1:]:
                agent.reset()

        self.rewards = {...}

        for e in range(self.nepisodes):
            episode = self.run_episode(agents, train_mode = True)
            norm_r = self.average_reward(episode)

            if verbose:
                print('train epoch:', e, 'avg steps:', episode.steps.float().mean().item(), 'avg reward:', norm_r)
            agents[1].update(episode)















