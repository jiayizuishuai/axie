import numpy as np
import os
import yaml
from collections import Counter


class Axie_Feature(object):
    def __init__(self, config):
        self.game_axie_num = 3
        self.game_minion_num = 5
        self.game_position_num = 6
        self.game_card_num = 370 # total cards num of game, always larger than the num of game cards
        self.game_rune_num = 50 # total runes num of game
        self.game_buff_num = 40 # total buffs num of game
        self.game_minion_type_num = 15 # total minions num of game, minions are summoned by axies
        self.game_secret_type_num = 10 # total types of secret cards. The type is based on the strategy
        self.game_card_type_num = 10 # total types of cards. e.g. attack card, skill card, power card, secret card ...
        self.game_card_tag_num = 20 # total tags num of cards. e.g. Banish, Retain, Innate, Ethereal, scry ...
        self.game_max_one_stack_buff_num = 20 # total max one stack buff
        self.game_max_inf_stack_buff_num = 20 # total max infinity stacks buff

        self.axie_max_hp = None
        self.axie_max_shield = 100
        self.axie_max_handcard = 5
        self.axie_max_energy = 3

        self.battle_max_round = 14
        self.battle_max_draw_pile = 18
        self.battle_max_banish_pile = 18
        self.battle_max_discard_pile = 18

        with open(os.path.join(os.path.dirname(__file__),'config', 'card2id.yaml'), "r") as f:
            self.card2id = yaml.load(f.read(), Loader=yaml.SafeLoader)

        self.duplicate_card = ['Anemone', 'Nimo', 'Puppy', 'Foxy', 'Nut Cracker', 'Puff', 'Nut Cracker', 'Little Owl', 'Peace Maker', 'Little Owl', 'Leaf Bug', 'Kotaro', 'Tiny Dino']

        self.buff_id = {'Bleed': 0, 'Bubble': 1, 'BubbleBomb': 2, 'Cleanser': 3, 'DamageBoost': 4,
                        'Disarmed': 5, 'Doubt': 6, 'Fear': 7, 'Feather': 8, 'Fragile': 9,
                        'Fury': 10, 'HealBlock': 11, 'HealingBoost': 12, 'Hex': 13, 'Leaf': 14,
                        'Poison': 15, 'Rage': 16, 'ShieldBoost': 17, 'Silence': 18, 'Sleep': 19,
                        'Stealth': 20, 'Stunned': 21, 'Taunt': 22, 'Vulnerable': 23, 'Weak': 24,
                        'Meditate': 25, 'DeathMark': 26}

        self.buff_normalizer = {'Bleed': 1, 'Bubble': 3, 'BubbleBomb': 1, 'Cleanser': 1, 'DamageBoost': 30,
                                'Disarmed': 1, 'Doubt': 1, 'Fear': 1, 'Feather': 10, 'Fragile': 1,
                                'Fury': 1, 'HealBlock': 1, 'HealingBoost': 30, 'Hex': 1, 'Leaf': 5,
                                'Poison': 30, 'Rage': 10, 'ShieldBoost': 30, 'Silence': 1, 'Sleep': 1,
                                'Stealth': 1, 'Stunned': 1, 'Taunt': 1, 'Vulnerable': 1, 'Weak': 1,
                                'Meditate': 1, 'DeathMark': 1}

        self.minion_id = {'Little Robin': 0, 'Mushroom': 1, 'Trunk': 2, 'Clover': 3,
                          'Mavis': 4, 'Fruit Sloth': 5, 'True-Fan Hermit Crab': 6, 'Sparrow': 7}

        self.round_normalizer = 15
        self.hand_card_num_normalizer = 5
        self.energy_normalizer = 3
        self.draw_pile_normalizer = 18
        self.discard_pile_normalizer = 18
        self.banish_pile_normalizer = 18


    def get_axie_feature(self, info):
        player_0, player_1, battle = info

        player_0.axies.sort(key=lambda x:x.position)
        player_1.axies.sort(key=lambda x:x.position)

        player_0.init_axies.sort(key=lambda x: x.position)
        player_1.init_axies.sort(key=lambda x: x.position)

        # constant state
        ally_axie_cards = self.get_card_name(player_0)
        enemy_axie_cards = self.get_card_name(player_1)


        self.ally_axie_equipped_card = self._card2id(ally_axie_cards)
        # self.ally_axie_rune = np.zeros(self.game_rune_num, dtype=np.int8)

        self.enemy_axie_equipped_card = self._card2id(enemy_axie_cards)
        # self.enemy_axie_rune = np.zeros(self.game_rune_num, dtype=np.int8)

        # live state
        self.ally_axie_hp = np.zeros(self.game_axie_num, dtype=np.float32)
        self.ally_axie_shield = np.zeros(self.game_axie_num, dtype=np.int8)
        self.ally_axie_position = np.zeros([self.game_axie_num, self.game_position_num], dtype=np.int8)

        self.ally_minion_position = np.zeros([self.game_minion_num, self.game_position_num], dtype=np.int8)
        self.ally_minion_hp = np.zeros(self.game_minion_num, dtype=np.float32)
        self.ally_minion_shield = np.zeros(self.game_minion_num, dtype=np.float32)
        self.ally_minion_type = np.zeros([self.game_minion_num, self.game_minion_type_num], dtype=np.int8)


        self.enemy_axie_hp = np.zeros(self.game_axie_num, dtype=np.float32)
        self.enemy_axie_shield = np.zeros(self.game_axie_num, dtype=np.int8)
        self.enemy_axie_position = np.zeros([self.game_axie_num, self.game_position_num], dtype=np.int8)

        self.enemy_minion_position = np.zeros([self.game_minion_num, self.game_position_num], dtype=np.int8)
        self.enemy_minion_hp = np.zeros(self.game_minion_num, dtype=np.float32)
        self.enemy_minion_shield = np.zeros(self.game_minion_num, dtype=np.float32)
        self.enemy_minion_type = np.zeros([self.game_minion_num, self.game_minion_type_num], dtype=np.int8)

        # self.ally_card_pile = np.zeros(self.game_card_num, dtype=np.int8)
        # self.ally_discard_pile = np.zeros(self.game_card_num, dtype=np.int8)
        # self.ally_banish_pile = np.zeros(self.game_card_num, dtype=np.int8)
        # self.ally_hand_card_pile = np.zeros(self.game_card_num, dtype=np.int8)
        #
        # self.enemy_played_card = np.zeros(self.game_card_num, dtype=np.int8)

        self.round = battle.turn_controller.turn / self.round_normalizer
        self.hand_card_num = len(player_0.hand_cards) / self.hand_card_num_normalizer
        self.energy = player_0.energy / self.energy_normalizer
        self.draw_pile_num = len(player_0.draw_pile._cards) / self.draw_pile_normalizer
        self.banish_pile_num = len(player_0.banish_pile._cards) / self.banish_pile_normalizer
        self.discard_pile_num = len(player_0.discard_pile._cards) / self.discard_pile_normalizer

        self.nums = np.array([self.round,
                              self.hand_card_num,
                              self.energy,
                              self.draw_pile_num,
                              self.banish_pile_num,
                              self.discard_pile_num], dtype=np.float32)

        axie_counter = 0
        minion_counter = 0
        for index, axie in enumerate(player_0.positions):
            if (axie is None):
                continue

            if (axie.is_summoned):
                self.ally_minion_hp[minion_counter] = float(axie.hp) / axie.max_hp
                self.ally_minion_shield[minion_counter] = float(axie.shield) / self.axie_max_shield
                self.ally_minion_position[minion_counter, axie.position] = 1
                self.ally_minion_type[minion_counter, self.minion_id[axie.name]] = 1

                minion_counter += 1
            else:
                self.ally_axie_hp[axie_counter] = float(axie.hp) / axie.max_hp
                self.ally_axie_shield[axie_counter] = float(axie.shield) / self.axie_max_shield
                self.ally_axie_position[axie_counter, axie.position] = 1
                axie_counter += 1


        self.ally_axie_buff = self._get_buff(player_0)


        axie_counter = 0
        minion_counter = 0
        for index, axie in enumerate(player_1.positions):
            if (axie is None):
                continue

            if (axie.is_summoned):
                self.enemy_minion_hp[minion_counter] = float(axie.hp) / axie.max_hp
                self.enemy_minion_shield[minion_counter] = float(axie.shield) / self.axie_max_shield
                self.enemy_minion_position[minion_counter, axie.position] = 1
                self.enemy_minion_type[minion_counter, self.minion_id[axie.name]] = 1
                minion_counter += 1
            else:
                self.enemy_axie_hp[axie_counter] = float(axie.hp) / axie.max_hp
                self.enemy_axie_shield[axie_counter] = float(axie.shield) / self.axie_max_shield
                self.enemy_axie_position[axie_counter, axie.position] = 1
                axie_counter += 1

        self.enemy_axie_buff = self._get_buff(player_1)

        self.ally_minion_position = self.ally_minion_position.flatten()
        self.ally_minion_type = self.ally_minion_type.flatten()
        self.ally_axie_position = self.ally_axie_position.flatten()
        self.ally_axie_buff = self.ally_axie_buff.flatten()
        self.enemy_minion_position = self.enemy_minion_position.flatten()
        self.enemy_minion_type = self.enemy_minion_type.flatten()
        self.enemy_axie_position = self.enemy_axie_position.flatten()
        self.enemy_axie_buff = self.enemy_axie_buff.flatten()


        return np.concatenate([self.nums,

                               self.ally_axie_equipped_card,
                               self.enemy_axie_equipped_card,

                               self.ally_minion_hp,
                               self.ally_minion_shield,
                               self.ally_minion_position,
                               self.ally_minion_type,
                               self.ally_axie_hp,
                               self.ally_axie_shield,
                               self.ally_axie_position,
                               self.ally_axie_buff,

                               self.enemy_minion_hp,
                               self.enemy_minion_shield,
                               self.enemy_minion_position,
                               self.enemy_minion_type,
                               self.enemy_axie_hp,
                               self.enemy_axie_shield,
                               self.enemy_axie_position,
                               self.enemy_axie_buff
                               ])

    def get_card_name(self, player):
        card_names = []
        for axie in player.init_axies:
            for card in axie.init_cards:
                if (card.name in self.duplicate_card):
                    card_names.append(card.name + '(' + card.part_type + ')')
                else:
                    card_names.append(card.name)

        return card_names

    def _card2id(self, cards):
        ids = np.zeros(self.game_card_num, dtype=np.int8)
        for card in cards:
            ids[self.card2id[card]] = 1

        return ids

    def _get_buff(self, player):
        buff_codes = np.zeros([self.game_axie_num, self.game_buff_num], dtype=np.float32)

        for axie_index, axie in enumerate(player.init_axies):
            buffs = axie.buffs.get_all_buffs()
            buff_counter = {}
            for buff in buffs:
                if (buff.name in self.buff_id.keys()):
                    if (buff.stackable):
                        buff_counter[buff.name] = buff.stacks
                    elif (buff.stackable == False):
                        buff_counter[buff.name] = 1

            for buff in buff_counter.keys():
                if (buff in self.buff_id.keys()):
                    buff_codes[axie_index, self.buff_id[buff]] = buff_counter[buff] / self.buff_normalizer[buff]

        return buff_codes


