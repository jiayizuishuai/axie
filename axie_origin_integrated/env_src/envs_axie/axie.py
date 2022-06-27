from simulator.base import TargetType
from simulator.simulator import Simulator
from simulator.entity_builder import loads_every_thing, random_player, build_player_by_id
from collections import OrderedDict
import numpy as np
import os

from .axie_feature import Axie_Feature
from .card_feature import Card_Feature
import yaml
import random


class Env_Axie(object):
    def __init__(self, config):
        self.config = config

        self.sim = Simulator()
        # self.sim.enable_print()

        self.players = [None, None]
        self.player_ids = [None, None]

        self.axie_feature = Axie_Feature(config)
        self.card_feature = Card_Feature(config)

        with open(os.path.join(config['path'], 'card2id.yaml'), "r") as f:
            self.card2id = yaml.load(f.read(), Loader=yaml.SafeLoader)


    def reset(self):
        self.sim.new_battle()

        positions = self.generate_random_positions()

        self.players[0] = build_player_by_id(self.extract_player_id(positions[0]))
        self.players[1] = build_player_by_id(self.extract_player_id(positions[1]))

        self.player_ids[0] = positions[0]
        self.player_ids[1] = positions[1]

        self.sim.add_player(self.players[0])
        self.sim.add_player(self.players[1])

    def step(self, action_card, action_target, player_index):

        self.players[player_index].place_card(action_card, action_target)

        obs = self.get_state([self.players[player_index], self.players[1 - player_index], self.sim.battle])
        reward = self.get_reward(player_index=player_index)
        done = self.get_done()
        flags = {}

        return obs, reward, done, flags

    def get_state(self, info):
        """
        get state can process the info to data which pass to server for prediction.
        the return includes:
            state: the players and battle's state without action
            legal_actions: the card object and card target
            encoded_legal_actions: the encoded data from legal_action

        obs = [the player this turn,
                the enemy player this turn,
                battle_info]
        """

        state = self.axie_feature.get_axie_feature(info)

        legal_actions, encoded_legal_actions = self.get_legal_actions(info)

        obs = OrderedDict({'state': state,
                           'legal_actions': legal_actions,
                           'encoded_legal_actions': encoded_legal_actions})

        return obs

    def get_reward(self, player_index):
        """
        # win, reward = 1
        # lose, reward = -1
        # draw, reward = 0
        """

        if (self.sim.is_finished()):
            if (self.players[player_index].alive):
                return 1
            elif (self.players[1 - player_index].alive):
                return -1

        return 0

    def get_done(self):
        return self.sim.is_finished()

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

    def extract_player_id(self, position):
        """
        name = model:{index of this card stack}_player-id:{player id in axie卡池数据-卡池模版}
        """

        return int(position.split('_')[1].split(':')[1])

    def generate_random_positions(self):
        return random.sample(self.config['positions'], 2)
