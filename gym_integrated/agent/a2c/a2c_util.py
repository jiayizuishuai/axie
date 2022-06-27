import collections
import random

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCriticNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCriticNet, self).__init__()

        self.input_size = int(np.product(observation_space.shape))
        self.output_size = action_space.n

        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc_pi = nn.Linear(256, self.output_size)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=1):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


def compute_target(v_final, r_lst, mask_lst, gamma):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return th.tensor(td_target[::-1]).float()


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
