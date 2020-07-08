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

 #TODO sigma is an important hyperparameter that needs to be tuned
class DRU:

    def __init__(self,sigma, comm_narrow = True, hard = False ):
        self.sigma = sigma
        self.comm_narrow = comm_narrow
        self.hard = hard

    def regularize (self, m ):

        m_reg = m + torch.randn(m.size()) * self.sigma
        if self.comm_narrow:
            m_reg = torch.sigmoid(m_reg)
        else:
            m_reg = torch.softmax(m_reg,0)

        return m_reg

    def discretize(self,m):

        if self.hard:
            if self.comm_narrow:
                return (m.gt(0.5).float() - 0.5).sign().float()
            else:
                m_ = torch.zeros_like(m)
                if m.dim() == 1:
                    _, idx = m.max(0)
                    m_[idx] = 1.
                elif m.dim() == 2:
                    _, idx = m.max(1)
                    for b in range(idx.size(0)):
                        m_[b,idx[b]] = 1.

                else:
                    raise ValueError('Inaccurate: {}'.format(m.size()))

                return m_
        else:
            scale = 2 * 20
            if self.comm_narrow:
                return torch.sigmoid((m.gt(0.5).float() - 0.5) * scale)
            else:
                return torch.softmax(m * scale, -1)

    def forward(self,m,train_mode):
        if train_mode:
            return self.regularize(m)
        else:
            return self.discretize(m)

