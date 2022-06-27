
from asyncio import base_tasks
from tabnanny import check
import typing
from collections import deque
from collections import namedtuple

import torch
from torch import nn
import numpy as np

from core.base_agent import BaseAgent
from agent.models import Model

def create_optimizers(args, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=args.learning_rate,
            momentum=args.momentum,
            eps=args.epsilon,
            alpha=args.alpha)
        optimizers[position] = optimizer
    return optimizers

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss

class DMCAgent(BaseAgent):
    def __init__(self, args):
        args = namedtuple('Struct', args.keys())(*args.values())
        self.args = args
        self.batch_size = args.batch_size
        self.model_name = ['landlord', 'landlord_up', 'landlord_down']

        self.learner_models = Model(device=args.train_device)
        self.optimizers = create_optimizers(args, self.learner_models)

        device = args.train_device
        if not device == "cpu":
            device = 'cuda:' + str(args.train_device)
        self.device = torch.device(device)

        self.infer_models = Model(device=args.infer_device)
        self.update_infer_model()

        self.buffer_list = {key: deque(maxlen=100) for key in self.model_name}
        self.mean_episode_return_buf = {p:deque(maxlen=args.buffer_size) for p in self.model_name}

    def store(self, data, role):
        self.buffer_list[role].append(data)

        # if self.check_update():
        #     self.train()

    def update_infer_model(self):
        for role in self.model_name:
            self.infer_models.get_model(role).load_state_dict(
                self.learner_models.get_model(role).state_dict())

    def get_action(self, position, z_batch, x_batch, flags):
        agent_output = self.infer_models.forward(
            position, z_batch, x_batch, flags=flags)
        _action_idx = int(agent_output['action'].cpu().detach().numpy())

        return _action_idx

    def get_init_hidden_state(self):
        return None

    def check_update(self):
        ready_list = [len(self.buffer_list[name]) > self.batch_size for name in self.model_name]
        return np.all(ready_list)

    def train(self):
        for role in self.model_name:
            _data_list = [self.buffer_list[role].popleft() for _ in range(self.batch_size)]
            key_list = _data_list[0].keys()

            batch = {
                key: torch.stack([torch.tensor(_data[key]) for _data in _data_list], dim=1)
                for key in key_list
            }

            obs_x_no_action = batch['obs_x_no_action'].to(self.device)
            obs_action = batch['obs_action'].to(self.device)
            obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
            obs_x = torch.flatten(obs_x, 0, 1)
            obs_z = torch.flatten(batch['obs_z'].to(self.device), 0, 1).float()
            target = torch.flatten(batch['target'].to(self.device), 0, 1)
            episode_returns = batch['episode_return'][batch['done']]
            self.mean_episode_return_buf[role].append(torch.mean(episode_returns).to(self.device))

            learner_model = self.learner_models.get_model(role)
                
            learner_outputs = learner_model(obs_z, obs_x, return_value=True)
            loss = compute_loss(learner_outputs['values'], target)
            stats = {
                'mean_episode_return_'+role: torch.mean(torch.stack([_r for _r in self.mean_episode_return_buf[role]])).item(),
                'loss_'+role: loss.item(),
            }
            
            self.optimizers[role].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(learner_model.parameters(), self.args.max_grad_norm)
            self.optimizers[role].step()

            print(stats)


