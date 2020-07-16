"""Implementation of a DIAL agent adapted to the HLE."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random


import numpy as np
import torch

from cnet import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = collections.namedtuple(
    'Transition', ['reward', 'observation', 'legal_actions', 'action', 'begin'])


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon parameter for the agent's e-greedy policy.

  Args:
    decay_period: float, the decay period for epsilon.
    step: Integer, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before training starts.
    epsilon: float, the epsilon value.

  Returns:
    A float, the linearly decaying epsilon value.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
  return epsilon + bonus


@gin.configurable
class DIALAgent(object):
    def __init__(self,
             observation_size=None,
               num_actions=None,
               num_players=None,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=500,
               update_period=4,
               stack_size=1,
               target_update_period=500,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.02,
               epsilon_eval=0.001,
               epsilon_decay_period=1000,
               learning_rate = 0.0005,
               momentum = 0.05,
               decay = 1.0):

     self.num_actions = num_actions
     self.observation_size = observation_size
     self.num_players = num_players
     self.gamma = gamma
     self.update_horizon = update_horizon
     self.cumulative_gamma = math.pow(gamma, update_horizon)
     self.min_replay_history = min_replay_history
     self.target_update_period = target_update_period
     self.epsilon_fn = epsilon_fn
     self.epsilon_train = epsilon_train
     self.epsilon_eval = epsilon_eval
     self.epsilon_decay_period = epsilon_decay_period
     self.update_period = update_period
     self.eval_mode = False
     self.training_steps = 0
     self.batch_staged = False


     self.policy_net = DRQNet(self.observation_size, self.num_actions)
     self.target_net = DRQNet(self.observation_size , self.num_actions)

    # need to create hidden state

    def begin_episode(self,current_player,legal_actions,observation):

        self._weight_update()
        self.action = self._select_action(observation, legal_actions)
        self._record_transition(current_player, 0, observation, legal_actions,
                        self.action, begin=True)

        return self.action

    def step(self, reward, current_player, legal_actions, observation):

        self._weight_update()
        self.action = self._select_action(observation,legal_actions)
        self._record_transition(current_player,reward,observation, legal_actions,
                              self.action)

        return self.action

    def end_episode (self, final_rewards):

        self._compute_loss(terminal_rewards = final_rewards)

    def _record_transition(self, current_player, reward, observation,
                           legal_actions, action, begin=False):
        self.transitions[current_player].append(
            Transition(reward, torch.tensor(observation, dtype=torch.uint8),
                       torch.tensor(legal_actions, dtype=torch.float32),
                       action, begin))

    def qs_for_actions(self,current_player):
       #here we take from the previous observation and action;
       #at time step one these will be zero

        self.observation = self.transitions[current_player -1].observation
        self.prev_action = self.transitions[current_player-1]
        q_vals = self.policy_net(self.observation , self.prev_action)

    def _select_action(self,observation, legal_actions):
        if self.eval_mode:
            epsilon = self.epsilon_eval
        else:
            epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                      self.min_replay_history, self.epsilon_train)

        if random.random() <= epsilon:
            # Choose a random action with probability epsilon.
            legal_action_indices = np.where(legal_actions == 0.0)
            return np.random.choice(legal_action_indices[0])
        else:
            action = q_vals(legal_actions).max(0) # TODO need to check this formula
            assert legal_actions[action] == 0.0, "Expected legal actions"
            return action




    #can only be called @end of episode
    def _compute_loss(self, terminal_rewards,num_players):
      for player in range(self.num_players):
          num_transitions = len(self.transitions[player])

          for index, transition in enumerate(self.transitions[player]):
              td_action = 0
              r_t = self.transitions[player][index].reward
              q_values = policy_net(self.transitions[player][index].observation , self.transitions.[player][index].whatever)
              #q_a_t is just the action value selected from q_values above
              q_a_t = q_values * action selected for that time step {obtain via replay buffer or transitions}

              t_q_values = target_net(same transsitions as policy net, but add +1 )
              q_a_max_t = action value from target t_q

              final_transition = index == num_transitions -1
              if final_transition:
                  reward = terminal_rewards[player]
                  td_action = reward - q_a_t
              else:
                  reward = terminal_rewards[player]
                  td_action = (reward + self.gamma(q_a_max_t)) - q_a_t

              loss_t = td_action ** 2
              total_loss = total_loss + loss_t

      loss = total_loss.sum()
      self.loss = loss / num_players
      self.transitions[player] = []

    def compute_episode_loss(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        clip_grad_norm_(parameters=self.policy_net.get_params(),max_norm=0)
        self.optimer.step()
      

  def _weight_update(self):
      if self.training_steps == self.target_update_period == 0:
          self.target_net.load_state_dict(self.policy_net.state_dict())







  def _train_step(self):
      #compute a loss for all steps





def end_episode(self, final_rewards):

    self._post_transitions(terminal_rewards=final_rewards)

##need to create - going through each agent,














