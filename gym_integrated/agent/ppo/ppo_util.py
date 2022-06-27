import collections
import random

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PPONet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(PPONet, self).__init__()

        self.input_size = int(np.product(observation_space.shape))
        self.output_size = action_space.n

        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc_pi = nn.Linear(256, self.output_size)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


class EpisdoeReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer_limit = buffer_limit
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, episode):
        self.buffer.append(episode)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        return mini_batch

    def size(self):
        return len(self.buffer)

    def clean(self):
        self.buffer = collections.deque(maxlen=self.buffer_limit)
