
from .models import Model
import torch

from env_src.envs_axie.axie_feature import Axie_Feature
from env_src.envs_axie.card_feature import Card_Feature
from simulator.base import *
from collections import OrderedDict

from torch import nn
from collections import deque
import numpy as np
import os
import time

class BaseAgent:
    def store(self, data):
        raise NotImplementedError(f"store func must implement")

    def update_infer_model(self):
        raise NotImplementedError(f"update_infer_model func must implement")

    def get_action(self, obs, hidden_state, infer_model_id=0):
        raise NotImplementedError(f"get_action func must implement")

    def get_init_hidden_state(self):
        raise NotImplementedError(f"get_init_hidden_state func must implement")

    def check_update(self):
        raise NotImplementedError(f"check_update func must implement")

    def train(self):
        raise NotImplementedError(f"train func must implement")

    def model_save(self):
        raise NotImplementedError(f"model_save func must implement")

def create_optimizers(args, learner_model, positions):
    """
    Create three optimizers for the three positions
    """
    positions = positions
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

class DMCV3Agent(BaseAgent):
    def __init__(self, args, data_queue_dict_list, model_queue_dict_list):
        self.args = args
        self.batch_size = args.batch_size
        self.model_names = args.model_names

        self.data_queue_dict_list = data_queue_dict_list
        self.model_queue_dict_list = model_queue_dict_list

        self.config = {'config_path': os.path.join(os.path.abspath(os.path.dirname(__file__)), '../env_src/envs_axie/config')}
        self.axie_feature = Axie_Feature(self.config)
        self.card_feature = Card_Feature(self.config)

        self.model_update_stats = {}
        for model_name in self.model_names:
            self.model_update_stats[model_name] = \
                {
                    'frames': 0,
                    'mean_episode_return_' + model_name: 0.0,
                    'loss_' + model_name: 0.0
                }

        device = args.train_device

        if not device == "cpu":
            device = 'cuda:' + str(args.train_device)
        self.device = torch.device(device)

        self.learner_models = Model(device=args.train_device, model_names=self.model_names)
        self.optimizers = create_optimizers(args, self.learner_models, self.model_names)

        if (args.load_model):
            self.model_load_dirs = {}
            for model_name in self.model_names:
                self.model_load_dirs[model_name] = self.args.model_load_dir + model_name + '/'
            self.load_models()

        self.update_infer_model(ready_list=[True for _ in self.model_names])

        self.buffer_list = {key: deque(maxlen=args.buffer_size) for key in self.model_names}
        self.mean_episode_return_buf = {p:deque(maxlen=args.buffer_size) for p in self.model_names}

        self.model_save_dirs = {}
        for model_name in self.model_names:
            self.model_save_dirs[model_name] = self.args.model_save_dir + model_name + '/'

            if (not os.path.exists(self.model_save_dirs[model_name])):
                os.makedirs(self.model_save_dirs[model_name])

        assert args.buffer_size > args.batch_size, "Buffer size must be larger than batch size!"

        self.log_file = "./log_file.txt"



    def store(self):
        for q_dict in self.data_queue_dict_list:
            for model_name in self.model_names:
                 while not q_dict[model_name].empty():
                    data = q_dict[model_name].get()
                    self.buffer_list[model_name].append(data)


    def update_infer_model(self, ready_list):
        for index, model_name in enumerate(self.model_names):
            if (ready_list[index] == False):
                continue

            for q_dict in self.model_queue_dict_list:
                    q_dict[model_name].put(self.learner_models.get_model(model_name).state_dict())

    def check_update(self):
        # temp = []
        # for model_name in self.model_names:
        #     temp.append(len(self.buffer_list[model_name]))
        #
        # print("Buffer size: {}".format(temp))

        ready_list = [len(self.buffer_list[model_name]) > self.batch_size for model_name in self.model_names]
        return ready_list

    def model_save(self):
        for model_name in self.learner_models.model_names:
            checkpoint_path = self.model_save_dirs[model_name] + 'model.tar'
            torch.save({'model_state_dict': self.learner_models.models[model_name].state_dict(),
                        'optimizer_state_dict': self.optimizers[model_name].state_dict(),
                        'stats': self.model_update_stats[model_name]
                        }, checkpoint_path)

            model_weight_path = self.model_save_dirs[model_name] + 'model.pth'
            torch.save(self.learner_models.models[model_name], model_weight_path)

        print("Model save finished!")


    def train(self, ready_list):
        for index, model_name in enumerate(self.model_names):
            if (ready_list[index] == False):
                continue

            _data_list = [self.buffer_list[model_name].popleft() for _ in range(self.batch_size)]
            key_list = _data_list[0].keys()

            batch_length = len(_data_list[0]['done'])

            batch = {
                key: torch.stack([torch.tensor(_data[key]) for _data in _data_list], dim=1)
                for key in key_list
            }

            obs_x_no_action = batch['obs_x_no_action'].to(self.device)
            obs_action = batch['obs_action'].to(self.device)
            obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
            obs_x = torch.flatten(obs_x, 0, 1)
            target = torch.flatten(batch['target'].to(self.device), 0, 1)
            episode_returns = batch['episode_return'][batch['done']]
            self.mean_episode_return_buf[model_name].append(torch.mean(episode_returns).to(self.device))

            learner_model = self.learner_models.get_model(model_name)

            learner_outputs = learner_model(obs_x, return_value=True)
            loss = compute_loss(learner_outputs['values'], target)

            stats = {
                'frames': self.model_update_stats[model_name]['frames'] + batch_length * self.batch_size,
                'mean_episode_return': torch.mean(
                    torch.stack([_r for _r in self.mean_episode_return_buf[model_name]])).item(),
                'loss': loss.item(),
            }

            self.model_update_stats[model_name] = stats

            self.optimizers[model_name].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(learner_model.parameters(), self.args.max_grad_norm)
            self.optimizers[model_name].step()

        if (int(time.time()) % self.args.log_interval == 0):
            for model_name in self.model_names:
                print('{} stats: '.format(model_name), self.model_update_stats[model_name])


            print("______________")

    def load_models(self):
        for model_name in self.model_names:
            checkpoint_states = torch.load(self.model_load_dirs[model_name] + "model.tar",
                                           map_location=('cpu')
                                           )

            self.learner_models.models[model_name].load_state_dict(checkpoint_states['model_state_dict'])
            self.optimizers[model_name].load_state_dict(checkpoint_states['optimizer_state_dict'])
            self.model_update_stats[model_name] = checkpoint_states['stats']

