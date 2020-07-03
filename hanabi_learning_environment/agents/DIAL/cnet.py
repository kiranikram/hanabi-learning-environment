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

### C-net DRQN

class CNet(nn.Module):
    def __init__(self,
                 num_players = None):


