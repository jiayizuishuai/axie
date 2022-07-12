import os

import torch
import numpy as np

from env_src.envs_axie.axie_feature import Axie_Feature
from env_src.envs_axie.card_feature import Card_Feature
from simulator.base import *
from collections import OrderedDict

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


from .models import Model


class DMCV3InferAgent(BaseAgent):
    def __init__(self, args, data_queue_dict, model_queue_dict):
        self.args = args
        self.model_names = args.model_names

        self.data_queue_dict = data_queue_dict
        self.model_queue_dict = model_queue_dict

        self.config = {'config_path': os.path.join(os.path.abspath(os.path.dirname(__file__)), '../env_src/envs_axie/config')}
        self.axie_feature = Axie_Feature(self.config)
        self.card_feature = Card_Feature(self.config)

        device = args.infer_device

        if not device == "cpu":
            device = 'cuda:' + str(args.train_device)
        self.device = torch.device(device)

        self.infer_models = Model(device=args.infer_device, model_names=self.model_names)

        for model_name in self.model_names:
            # 这里将会阻塞住，直到更新参数过来
            parameter = self.model_queue_dict[model_name].get()
            self.update_infer_model(model_name, parameter)

    def store(self, data, model_name):
        # TODO：是否需要攒一波？
        self.data_queue_dict[model_name].put(data)

    def update_infer_model(self, model_name, parameter):
        self.infer_models.get_model(model_name).load_state_dict(parameter)

    def get_action(self, position, data, flags):

        if (flags['data_type'] == 'simulator'):
            state = self.axie_feature.get_axie_feature(sim.battle)
            legal_actions, encoded_legal_actions = self.get_legal_actions(sim.battle)

            x_batch = OrderedDict({'state': state,
                                   'legal_actions': legal_actions,
                                   'encoded_legal_actions': encoded_legal_actions})

        elif (flags['data_type'] == 'code'):
            x_batch = OrderedDict({'state': np.array(data['state'], dtype=np.float32),
                                   'encoded_legal_actions': np.array(data['encoded_legal_actions'], dtype=np.float32)})


        agent_output = self.infer_models.forward(position, x_batch, flags=flags)
        _action_idx = int(agent_output['action'].cpu().detach().numpy())


        if (flags['data_type'] == 'simulator'):
            response = {'action': x_batch['legal_actions'][_action_idx],
                        'encoded_action': x_batch['encoded_legal_actions'][_action_idx]}

        elif (flags['data_type'] == 'code'):
            response = {'action_idx': _action_idx,
                        'encoded_action': x_batch['encoded_legal_actions'][_action_idx].tolist()}

        # TODO：是否有更优雅的方式检查模型是否更新？
        for model_name in self.model_names:
             while not self.model_queue_dict[model_name].empty():
                parameter = self.model_queue_dict[model_name].get()
                self.update_infer_model(model_name, parameter)

        return response

    def get_init_hidden_state(self):
        return None

    def get_legal_actions(self, battle):
        player = battle.current_player
        enemy_player = player.enemy

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
            encoded_legal_actions.append(self.card_feature._card2array(battle, action))


        return legal_actions, encoded_legal_actions