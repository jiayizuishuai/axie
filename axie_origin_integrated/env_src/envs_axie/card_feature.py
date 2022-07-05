from simulator.base import *
from simulator.simulator import Simulator
from simulator.entity_builder import *

import numpy as np
import os
import yaml


class EffectRecorder(Component):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.inst: Optional["Card"] = None
        self.damage_infos = []

    def on_add(self, inst: "Card"):
        super().on_add(inst)
        inst.add_event_listener(CardEvent.OnDamageCalculated, self.on_damage_calculated)

    def on_remove(self, inst: "Card"):
        inst.remove_event_listener(CardEvent.OnDamageCalculated, self.on_damage_calculated)
        super().on_remove(inst)

    def on_damage_calculated(self, inst: "Card", damage: int, target: "Axie"):
        self.damage_infos.append(
            {
                "damage": damage,
                "target_snapshot": target.base_info() if target else None,
                "target": target,
                "self_snapshot": inst.axie.base_info()
            }
        )


class Card_Feature(object):
    def __init__(self, config):
        self.game_position_num = 6
        self.game_card_num = 370 # total cards num of game, always larger than the num of game cards

        with open(os.path.join(os.path.dirname(__file__), 'config', 'card2id.yaml'), "r") as f:
            self.card2id = yaml.load(f.read(), Loader=yaml.SafeLoader)

        self.duplicate_card = ['Anemone', 'Nimo', 'Puppy', 'Foxy', 'Nut Cracker', 'Puff', 'Nut Cracker', 'Little Owl', 'Peace Maker', 'Little Owl', 'Leaf Bug', 'Kotaro', 'Tiny Dino']


    def _card_feature2array(self, card, target, battle, player):

        def choose_first_target(card):
            target_type = card.target_type
            if target_type == TargetType.Auto:
                return -1
            elif target_type == TargetType.Ally:
                for idx, axie in enumerate(player.positions):
                    if axie:
                        return idx
            elif target_type == TargetType.Enemy:
                for idx, axie in enumerate(player.enemy.positions):
                    if axie and axie.alive and axie.targetable:
                        return idx
            elif target_type == TargetType.SummonPosition:
                for idx, axie in enumerate(player.positions):
                    if not axie:
                        return idx


        with player.battle.incognito_mode() as battle_copy:
            cur_player = battle_copy.current_player
            _cards = cur_player.hand_cards
            for card in _cards:
                if card.where != PileType.Hand:
                    # transformed or something else
                    continue
                if card.has_component("Revenge"):
                    print(card)
                if card.name.startswith("The"):
                    print(card)
                if not card.can_play:
                    continue
                if not cur_player.enemy.alive:
                    # check if enemy is alive (any axie alive)
                    break
                effect = EffectRecorder()
                card.add_component(effect)
                position = choose_first_target(card)
                cur_player.place_card(card, position)
                # print(f"{cur_player} place {card.name} to {position} on {cur_player.battle.turn}")
                card.remove_component(effect)
                print(effect.damage_infos)
                print("==========")

    def _card2array(self, info, action):
        player_0, player_1, battle = info
        card, position = action

        card_array = np.zeros(self.game_card_num, dtype=np.int8)
        ally_array = np.zeros(self.game_position_num, dtype=np.int8)
        enemy_array = np.zeros(self.game_position_num, dtype=np.int8)

        if (card != 'end_turn'):
            card_array[self.card2id[self.get_card_name(card)]] = 1

            if (card.target_type == TargetType.Ally):
                ally_array[position] = 1

            if (card.target_type == TargetType.Enemy):
                enemy_array[position] = 1

        return np.concatenate([card_array, ally_array, enemy_array], axis=0)

    def get_card_name(self, card):
        card_name = None
        if (card.name in self.duplicate_card):
            card_name = card.name + '(' + card.part_type + ')'
        else:
            card_name = card.name

        return card_name














