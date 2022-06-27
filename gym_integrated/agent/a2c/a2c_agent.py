import torch as th
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from core.base_agent import BaseAgent
from agent.a2c.a2c_util import ActorCriticNet, compute_target, EpisdoeReplayBuffer


class A2CAgent(BaseAgent):
    def __init__(self, args, observation_space, action_space):
        self.learning_rate = args['learning_rate']
        self.gamma = args['gamma']
        self.buffer_limit = args['buffer_limit']
        self.update_episode_interval = args['update_episode_interval']

        self.ac_model = ActorCriticNet(observation_space, action_space)
        self.infer_model = ActorCriticNet(observation_space, action_space)

        self.update_infer_model()

        self.optimizer = optim.Adam(
            self.ac_model.parameters(), lr=self.learning_rate)
        self.buffer = EpisdoeReplayBuffer(buffer_limit=self.buffer_limit)

    def store(self, data):
        obs_lst, act_lst, r_lst, mask_lst = list(), list(), list(), list()

        for i in range(0, len(data)-1, 4):
            obs, action, reward, info, next_obs = data[i:i+5]
            obs_lst.append(obs)
            act_lst.append(action)
            r_lst.append(reward/100.0)
            mask_lst.append(0 if i + 5 == len(data) else 1)

        self.buffer.put([obs_lst, act_lst, r_lst, mask_lst])

    def update_infer_model(self):
        self.infer_model.load_state_dict(self.ac_model.state_dict())

    def get_action(self, obs, hidden_state):
        prob = self.infer_model.pi(th.from_numpy(obs).float(), softmax_dim=-1)
        a = Categorical(prob).sample().numpy()
        return a, None

    def get_init_hidden_state(self):
        return None

    def check_update(self):
        return self.buffer.size() > self.update_episode_interval

    def train(self):
        # step 1 将不同长度的episode处理成可以用的
        total_obs_lst, total_act_lst, total_r_lst, total_mask_lst = list(), list(), list(), list()
        episode_batch = self.buffer.sample(self.buffer.size())
        for episode in episode_batch:
            obs_lst, act_lst, r_lst, mask_lst = episode
            total_obs_lst.extend(obs_lst)
            total_act_lst.extend(act_lst)
            total_r_lst.extend(r_lst)
            total_mask_lst.extend(mask_lst)
        self.buffer.clean()

        # step 2
        v_final = np.zeros(shape=[1], dtype=np.float32)
        td_target = compute_target(
            v_final, total_r_lst, total_mask_lst, self.gamma)
        td_target_vec = td_target.reshape(-1)
        s_vec = th.tensor(total_obs_lst).float(
        ).reshape(-1, self.ac_model.input_size)
        a_vec = th.tensor(np.array(total_act_lst)).reshape(-1).unsqueeze(1)
        advantage = td_target_vec - self.ac_model.v(s_vec).reshape(-1)

        pi = self.ac_model.pi(s_vec, softmax_dim=1)
        pi_a = pi.gather(1, a_vec).reshape(-1)
        loss = -(th.log(pi_a) * advantage.detach()).mean() + \
            F.smooth_l1_loss(self.ac_model.v(s_vec).reshape(-1), td_target_vec)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
