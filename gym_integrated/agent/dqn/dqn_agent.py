import copy

import torch as th
import torch.optim as optim
import torch.nn.functional as F

from core.base_agent import BaseAgent
from agent.dqn.dqn_util import Qnet, ReplayBuffer


class DQNAgent(BaseAgent):
    def __init__(self, args, observation_space, action_space):
        self.learning_rate = args['learning_rate']  # 0.0005
        self.gamma = args['gamma']  # 0.98
        self.buffer_limit = args['buffer_limit']  # 50000
        self.batch_size = args['batch_size']  # 32
        self.train_num_i = args['train_num_i']  # 10
        self.epsilon = args['epsilon']  # 0.08

        self.buffer = ReplayBuffer(buffer_limit=self.buffer_limit)
        self.q = Qnet(observation_space, action_space)
        self.q_target = Qnet(observation_space, action_space)
        self.q_target.load_state_dict(self.q.state_dict())

        self.q_infer = Qnet(observation_space, action_space)
        self.update_infer_model()

        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)
        self.q_update = 0

    def store(self, data):
        for i in range(0, len(data)-1, 4):
            obs, action, reward, info, next_obs = data[i:i+5]
            self.buffer.put((obs, action, reward/100.0, next_obs,
                             0.0 if i + 5 == len(data) else 1.0))

    def update_infer_model(self):
        self.q_infer.load_state_dict(self.q.state_dict())

    def get_action(self, obs, hidden_state):
        a = self.q_infer.sample_action(
            th.from_numpy(obs).float(), self.epsilon)
        return a, None

    def get_init_hidden_state(self):
        return None

    def check_update(self):
        return self.buffer.size() > 2000  # self.batch_size

    def train(self):
        self.q_update += 1
        print('training!')
        for _ in range(self.train_num_i):
            s, a, r, s_prime, done_mask = self.buffer.sample(self.batch_size)

            q_out = self.q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + self.gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.q_update == 20:
            self.q_target.load_state_dict(self.q.state_dict())
            self.q_update = 0
