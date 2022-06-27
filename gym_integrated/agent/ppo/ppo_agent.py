import torch as th
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from core.base_agent import BaseAgent
from agent.ppo.ppo_util import PPONet, EpisdoeReplayBuffer


class PPOAgent(BaseAgent):
    def __init__(self, args, observation_space, action_space):
        self.learning_rate = args['learning_rate']
        self.gamma = args['gamma']
        self.lmbda = args['lmbda']
        self.eps_clip = args['eps_clip']
        self.K_epoch = args['K_epoch']
        self.buffer_limit = args['buffer_limit']
        self.update_episode_interval = args['update_episode_interval']

        self.ppo_model = PPONet(observation_space, action_space)
        self.infer_model = PPONet(observation_space, action_space)

        self.update_infer_model()

        self.optimizer = optim.Adam(
            self.ppo_model.parameters(), lr=self.learning_rate)
        self.buffer = EpisdoeReplayBuffer(buffer_limit=self.buffer_limit)

    def store(self, data):
        obs_lst, act_lst, act_prob_list, r_lst, mask_lst, next_obs_lst = list(
        ), list(), list(), list(), list(), list()

        for i in range(0, len(data)-1, 5):
            obs, action, action_prob, reward, info, next_obs = data[i:i+6]
            obs_lst.append(obs)
            act_lst.append([action])
            act_prob_list.append([action_prob])
            r_lst.append([reward/100.0])
            mask_lst.append([0 if i + 6 == len(data) else 1])
            next_obs_lst.append(next_obs)

        self.buffer.put([obs_lst, act_lst, act_prob_list,
                         r_lst, mask_lst, next_obs_lst])

    def update_infer_model(self):
        self.infer_model.load_state_dict(self.ppo_model.state_dict())

    def get_action(self, obs, hidden_state):
        prob = self.infer_model.pi(th.from_numpy(obs).float(), softmax_dim=-1)
        a = Categorical(prob).sample().numpy()
        return a, None, prob[a].item()

    def get_init_hidden_state(self):
        return None

    def check_update(self):
        return self.buffer.size() >= self.update_episode_interval

    def train(self):
        total_obs_lst, total_act_lst, total_act_prob_lst, total_r_lst, total_mask_lst, total_next_obs_lst = list(
        ), list(), list(), list(), list(), list()

        episode_batch = self.buffer.sample(self.buffer.size())
        for episode in episode_batch:
            obs_lst, act_lst, act_prob_list, r_lst, mask_lst, next_obs_lst = episode
            total_obs_lst.extend(obs_lst)
            total_act_lst.extend(act_lst)
            total_act_prob_lst.extend(act_prob_list)
            total_r_lst.extend(r_lst)
            total_mask_lst.extend(mask_lst)
            total_next_obs_lst.extend(next_obs_lst)
        self.buffer.clean()

        total_obs = th.tensor(total_obs_lst, dtype=th.float)
        total_act = th.tensor(np.array(total_act_lst))
        total_act_prob = th.tensor(total_act_prob_lst)
        total_r = th.tensor(total_r_lst)
        total_mask = th.tensor(total_mask_lst, dtype=th.float)
        total_next_obs = th.tensor(total_next_obs_lst, dtype=th.float)

        for i in range(self.K_epoch):
            td_target = total_r + self.gamma * \
                self.ppo_model.v(total_next_obs) * total_mask
            delta = td_target.detach() - self.ppo_model.v(total_obs)
            delta = delta.detach().numpy()
            total_mask_np = total_mask.detach().numpy()

            advantage_lst = []
            advantage = 0.0

            for delta_t, done_t in zip(delta[::-1], total_mask_np[::-1]):
                if done_t:
                    advantage = 0.0
                advantage = self.gamma * self.lmbda * advantage + delta_t
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = th.tensor(advantage_lst, dtype=th.float)

            pi = self.ppo_model.pi(total_obs, softmax_dim=1)

            pi_a = pi.gather(1, total_act)
            # a/b == exp(log(a)-log(b))
            ratio = th.exp(th.log(pi_a) - th.log(total_act_prob))

            surr1 = ratio * advantage
            surr2 = th.clamp(ratio, 1-self.eps_clip, 1 +
                             self.eps_clip) * advantage
            loss = -th.min(surr1, surr2) + \
                F.smooth_l1_loss(self.ppo_model.v(
                    total_obs), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
