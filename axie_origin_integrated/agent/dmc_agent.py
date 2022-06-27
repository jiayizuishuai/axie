from core.base_agent import BaseAgent
from .models import Model
import torch

from axie_origin_integrated.env_src.envs_axie.axie_feature import Axie_Feature
from axie_origin_integrated.env_src.envs_axie.card_feature import Card_Feature
from simulator.base import *
from collections import OrderedDict

from torch import nn
from collections import deque
import numpy as np
import os

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
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.model_names = args.model_names.split('|')

        self.config = {'path': '/Users/francis/Public/francis/project/rct-env-driven-middleware/axie_origin_integrated/env_src/envs_axie/config'}
        self.axie_feature = Axie_Feature(self.config)
        self.card_feature = Card_Feature(self.config)

        device = args.train_device

        if not device == "cpu":
            device = 'cuda:' + str(args.train_device)
        self.device = torch.device(device)

        self.model_load_dirs = {}
        for model_name in self.model_names:
            self.model_load_dirs[model_name] = self.args.model_load_dir + model_name + '/'

        if (args.load_model == False):
            self.infer_models = Model(device=args.infer_device, model_names=self.model_names)
        else:
            self.infer_models = self.load_models()


        if (args.agent_type == 'train'):
            self.learner_models = Model(device=args.train_device, model_names=self.model_names)
            self.optimizers = create_optimizers(args, self.learner_models, self.model_names)
            self.update_infer_model()

            self.buffer_list = {key: deque(maxlen=args.buffer_size) for key in self.model_names}
            self.mean_episode_return_buf = {p:deque(maxlen=args.buffer_size) for p in self.model_names}

            self.model_save_dirs = {}
            for model_name in self.model_names:
                self.model_save_dirs[model_name] = self.args.model_save_dir + model_name + '/'

                if (not os.path.exists(self.model_save_dirs[model_name])):
                    os.makedirs(self.model_save_dirs[model_name])

            assert args.buffer_size > args.batch_size, "Buffer size must be larger than batch size!"


    def store(self, data, model_name):
        self.buffer_list[model_name].append(data)

    def update_infer_model(self):
        for model_name in self.model_names:
            self.infer_models.get_model(model_name).load_state_dict(
                self.learner_models.get_model(model_name).state_dict())

    def get_action(self, position, sim, flags):
        info = [sim.battle.current_player, sim.battle.current_player.enemy, sim.battle]

        state = self.axie_feature.get_axie_feature(info)

        legal_actions, encoded_legal_actions = self.get_legal_actions(info)

        x_batch = OrderedDict({'state': state,
                               'legal_actions': legal_actions,
                               'encoded_legal_actions': encoded_legal_actions})


        agent_output = self.infer_models.forward(position, x_batch, flags=flags)
        _action_idx = int(agent_output['action'].cpu().detach().numpy())

        response = {'action': x_batch['legal_actions'][_action_idx],
                    'encoded_action': x_batch['encoded_legal_actions'][_action_idx]}

        return response



    def check_update(self):
        temp = []
        for model_name in self.model_names:
            temp.append(len(self.buffer_list[model_name]))

        print(temp)

        ready_list = [len(self.buffer_list[model_name]) > self.batch_size for model_name in self.model_names]
        return np.all(ready_list)

    def model_save(self):
        for model_name in self.learner_models.model_names:
            checkpoint_path = self.model_save_dirs[model_name] + 'model.tar'
            torch.save({'model_state_dict': self.learner_models.models[model_name].state_dict(),
                        'optimizer_state_dict': self.optimizers[model_name].state_dict(),
                        'stats': 0,
                        'frames': 0}, checkpoint_path)

            print(checkpoint_path)

            model_weight_path = self.model_save_dirs[model_name] + 'model.pth'
            torch.save(self.learner_models.models[model_name], model_weight_path)

        print("Model save finished!")


    def train(self):
        for model_name in self.model_names:
            _data_list = [self.buffer_list[model_name].popleft() for _ in range(self.batch_size)]
            key_list = _data_list[0].keys()

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
                'mean_episode_return_' + model_name: torch.mean(
                    torch.stack([_r for _r in self.mean_episode_return_buf[model_name]])).item(),
                'loss_' + model_name: loss.item(),
            }

            self.optimizers[model_name].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(learner_model.parameters(), self.args.max_grad_norm)
            self.optimizers[model_name].step()

        print('model update finished!')

    def get_init_hidden_state(self):
        return None


    def get_legal_actions(self, info):
        player, enemy_player, battle = info
        cards = player.hand_cards

        legal_actions = [['end_turn', -1]]
        encoded_legal_actions = []

        # find the legal action card and add into list
        for card in cards:

            if (card.can_play == False):
                continue

            target_type = card.target_type

            if target_type == TargetType.Auto:
                legal_actions.append([card, -1])

            elif target_type == TargetType.Ally:
                for position, axie in enumerate(player.positions):
                    if axie:
                        legal_actions.append([card, position])

            elif target_type == TargetType.Enemy:
                for position, axie in enumerate(player.enemy.positions):
                    if axie:
                        legal_actions.append([card, position])

        # encode the legal action card
        for action in legal_actions:
            encoded_legal_actions.append(self.card_feature._card2array(info, action))


        return legal_actions, encoded_legal_actions

    def load_models(self):
        model = Model(device=self.args.infer_device, model_init=False)

        for model_name in self.model_names:
            model_path = self.model_load_dirs[model_name] + 'model.pth'
            if os.path.isfile(model_path):
                model.add_model(model=torch.load(model_path, map_location=self.device),
                                model_name=model_name)
            else:
                raise ('Not a Model File!')

        return model
