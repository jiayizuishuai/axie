from simulator.base import TargetType
from simulator.simulator import Simulator
from simulator.entity_builder import loads_every_thing, random_player, build_player_by_id
from collections import OrderedDict
import numpy as np
import pandas as pd
import os

from .axie_feature import Axie_Feature
from .card_feature import Card_Feature
import yaml
import random
from simulator.utils import card_effects
from simulator.base import *
import time

class Env_Axie(object):
    def __init__(self, config):
        self.config = config

        self.sim = Simulator()
        # self.sim.enable_print()

        self.players = [None, None]
        self.player_ids = [None, None]

        self.axie_feature = Axie_Feature(config)
        self.card_feature = Card_Feature(config)

        with open(os.path.join(config['config_path'], 'card2id.yaml'), "r") as f:
            self.card2id = yaml.load(f.read(), Loader=yaml.SafeLoader)

        self.positions = pd.read_csv(os.path.join(config['config_path'], 'positions_info.csv'))['positions'].values.tolist()


    def reset(self):
        self.sim.new_battle()

        #positions = self.generate_random_positions()

        self.players[0] = build_player_by_id(52)
        self.players[1] = build_player_by_id(52)

        self.player_ids[0] = 'model_main_player-id_52'
        self.player_ids[1] = 'model_0_player-id_52'

        self.sim.add_player(self.players[0])
        self.sim.add_player(self.players[1])

    def step(self, action_card, action_target, player_index):
        card = None

        for card_ in self.players[player_index].cards:
            if (card_.id == action_card.id):
                card = card_
                break

        if (not card):
            raise Exception("card not found!")

        flag = self.players[player_index].place_card(card, action_target)

        # obs = self.get_state([self.players[player_index], self.players[1 - player_index], self.sim.battle])
        # reward = self.get_reward(player_index=player_index)
        done = self.get_done()
        # flags = {}

        return None, None, done, None

    def get_state(self, list_type=False):
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

        state = self.axie_feature.get_axie_feature(self.sim.battle)
        legal_actions, encoded_legal_actions = self.get_legal_actions()

        #  下面是动作编码的获得
        #start = time.time()
        play_card_encodes = []
        for legal_action in legal_actions:
            action_card, action_target = legal_action
            #play_card_to_state=env.clone_step(action_card, action_target, current_player_index)
            #play_card_to_states.append(play_card_to_state)
            if action_card == 'end_turn':
                play_card_encode = np.zeros(241)
                play_card_encodes.append(play_card_encode)
            if action_card != 'end_turn':
                play_card_encode = self.get_action_encode(action_card, action_target)
                play_card_encodes.append(play_card_encode)

        encoded_legal_actions = play_card_encodes
        #print(time.time() - start)



        if (list_type==False):
            data = OrderedDict({'state': state,
                                'legal_actions': legal_actions,
                                'encoded_legal_actions': encoded_legal_actions})

            return data

        else:
            state_list = state.tolist()

            encoded_legal_actions_list = []
            for encoded_legal_action in encoded_legal_actions:
                encoded_legal_actions_list.append(encoded_legal_action.tolist())

            encoded_data = {'state': state_list,
                            'encoded_legal_actions': encoded_legal_actions_list}

            data = {'encoded_data': encoded_data,
                    'legal_actions': legal_actions}

            return data


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

    def get_legal_actions(self):
        player = self.sim.battle.current_player
        enemy_player = player.enemy

        cards = player.hand_cards

        legal_actions = []
        encoded_legal_actions = []

        # find the legal action card and add into list
        for card in cards:

            if (card.can_play == False):
                continue
            if card.where != PileType.Hand:
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

            elif target_type == TargetType.SummonPosition:
                for position, axie in enumerate(player.positions):
                    if not axie:
                        legal_actions.append([card, position])
        legal_actions.append(['end_turn', -1])
        # encode the legal action card
        for action in legal_actions:
            encoded_legal_actions.append(self.card_feature._card2array(self.sim.battle, action))


        return legal_actions, encoded_legal_actions

    def extract_player_id(self, position):
        """
        name = model:{index of this card stack}_player-id:{player id in axie卡池数据-卡池模版}
        """

        return int(position.split('id_')[1])

    def generate_random_positions(self):
        return random.sample(self.positions, 2)

    def debug_log(self):
        print("current battle round: {}".format(self.sim.battle.turn_controller.turn))

        current_player = self.sim.battle.current_player

        print("current player build id: {}".format(current_player.build_id))
        for axie in current_player.init_axies:
            print("Axie Hp: {} | Axie Max HP: {}".format(axie.hp, axie.max_hp))

        print("current enemy player build id: {}".format(current_player.enemy.build_id))
        for axie in current_player.enemy.init_axies:
            print("Axie Hp: {} | Axie Max HP: {}".format(axie.hp, axie.max_hp))


    def get_action_encode(self, action_card, action_target):
        #动作编码
        def action_encoder(effects, card):
            actions = []
            for i in range(8):

                small_action = np.zeros(70, dtype=np.float32)  # 0:指定目标，1：随机，2：全体
                if i == 0 and i in effects.ally:
                    small_action = np.zeros(70, dtype=np.float32)
                    small_action[0] = 0.1
                    ally_samll_action_encode(effects.ally[i], small_action)
                    actions.append(small_action)
                if i == 1 and i in effects.ally:
                    small_action = np.zeros(70, dtype=np.float32)
                    small_action[0] = 0.3
                    ally_samll_action_encode(effects.ally[i], small_action)
                    actions.append(small_action)
                if i == 2 and i in effects.ally:
                    small_action = np.zeros(70, dtype=np.float32)
                    small_action[0] = 0.5
                    ally_samll_action_encode(effects.ally[i], small_action)
                    actions.append(small_action)
                if i == 3 and i in effects.ally:
                    small_action = np.zeros(70, dtype=np.float32)
                    small_action[0] = 0.6
                    ally_samll_action_encode(effects.ally[i], small_action)
                    actions.append(small_action)
                if i == 4 and i in effects.ally:
                    small_action = np.zeros(70, dtype=np.float32)
                    small_action[0] = 0.8
                    ally_samll_action_encode(effects.ally[i], small_action)
                    actions.append(small_action)
                if i == 5 and i in effects.ally:
                    small_action = np.zeros(70, dtype=np.float32)
                    small_action[0] = 1.0
                    ally_samll_action_encode(effects.ally[i], small_action)
                    actions.append(small_action)
                if i == 6 and i in effects.ally:
                    small_action = np.zeros(70, dtype=np.float32)
                    small_action[1] = 1.0
                    ally_samll_action_encode(effects.ally[i], small_action)
                    actions.append(small_action)
                if i == 7 and i in effects.ally:
                    small_action = np.zeros(70, dtype=np.float32)
                    small_action[2] = 1.0
                    ally_samll_action_encode(effects.ally[i], small_action)
                    actions.append(small_action)
            for i in range(8):

                if i == 0 and i in effects.enemy:
                    small_action = np.zeros(70, dtype=np.float32)  # 0:指定目标，1：随机，2：全体
                    small_action[0] = -0.1
                    enemy_samll_action_encode(effects.enemy[i], small_action)
                    actions.append(small_action)
                if i == 1 and i in effects.enemy:
                    small_action = np.zeros(70, dtype=np.float32)  # 0:指定目标，1：随机，2：全体
                    small_action[0] = -0.3
                    enemy_samll_action_encode(effects.enemy[i], small_action)
                    actions.append(small_action)
                if i == 2 and i in effects.enemy:
                    small_action = np.zeros(70, dtype=np.float32)  # 0:指定目标，1：随机，2：全体
                    small_action[0] = -0.5
                    enemy_samll_action_encode(effects.enemy[i], small_action)
                    actions.append(small_action)
                if i == 3 and i in effects.enemy:
                    small_action = np.zeros(70, dtype=np.float32)  # 0:指定目标，1：随机，2：全体
                    small_action[0] = -0.6
                    enemy_samll_action_encode(effects.enemy[i], small_action)
                    actions.append(small_action)
                if i == 4 and i in effects.enemy:
                    small_action = np.zeros(70, dtype=np.float32)  # 0:指定目标，1：随机，2：全体
                    small_action[0] = -0.8
                    enemy_samll_action_encode(effects.enemy[i], small_action)
                    actions.append(small_action)
                if i == 5 and i in effects.enemy:
                    small_action = np.zeros(70, dtype=np.float32)  # 0:指定目标，1：随机，2：全体
                    small_action[0] = -1.0
                    enemy_samll_action_encode(effects.enemy[i], small_action)
                    actions.append(small_action)
                if i == 6 and i in effects.enemy:
                    small_action = np.zeros(70, dtype=np.float32)  # 0:指定目标，1：随机，2：全体
                    if len(effects.enemy[i].damages) == 1:
                        small_action[1] = -0.2
                    if len(effects.enemy[i].damages) == 2:
                        small_action[1] = -0.4
                    if len(effects.enemy[i].damages) == 3:
                        small_action[1] = -0.6
                    if len(effects.enemy[i].damages) == 4:
                        small_action[1] = -0.8
                    enemy_samll_action_encode(effects.enemy[i], small_action)
                    actions.append(small_action)
                if i == 7 and i in effects.enemy:
                    small_action = np.zeros(70, dtype=np.float32)  # 0:指定目标，1：随机，2：全体
                    small_action[2] = -1.0
                    enemy_samll_action_encode(effects.enemy[i], small_action)
                    actions.append(small_action)
            while len(actions) > 3:
                if len(actions) > 3:
                    print('原子动作大于3')
                    print(actions)
                    print((action_card.name))
                actions.pop()
            if len(actions) == 0:
                zero3 = np.zeros(210)
                actions.append(zero3)

            elif len(actions) == 1:
                zero2 = np.zeros(140)
                actions.append(zero2)
            elif len(actions) == 2:
                zero1 = np.zeros(70)
                actions.append(zero1)

            card_action = effect_to_card_action_encode(effects, card)
            actions.append(card_action)
            energy_used = np.zeros(1)
            energy_used[0] = effects.energy_used
            actions.append(energy_used)

            action_encode = np.concatenate(actions)
            action_encode = np.float32(action_encode)
            return action_encode

        def enemy_samll_action_encode(effect, small_action):
            if len(effect.damages) != 0:
                small_action[3] = effect.damages[0] / 500
            if len(effect.heals) != 0:
                small_action[4] = effect.heals[0] / 500
            if len(effect.shields) != 0:
                small_action[5] = effect.shields[0] / 500

            if 'Bleed' in effect.buffs:
                small_action[20] = effect.buffs['Bleed'] / 10
            if 'Bubble' in effect.buffs:
                small_action[14] = effect.buffs['Bubble'] / 10
            if 'BubbleBomb' in effect.buffs:
                small_action[8] = effect.buffs['BubbleBomb'] / 1.0
            if 'Cleanser' in effect.buffs:
                small_action[11] = effect.buffs['Cleanser'] / 10
            if 'DamageBoost' in effect.buffs:
                small_action[12] = effect.buffs['DamageBoost'] / 10
            if 'Disarmed' in effect.buffs:
                small_action[22] = effect.buffs['Disarmed'] / 10
            if 'Doubt' in effect.buffs:
                small_action[23] = effect.buffs['Doubt'] / 10
            if 'Fear' in effect.buffs:
                small_action[24] = effect.buffs['Fear'] / 10
            if 'Feather' in effect.buffs:
                small_action[13] = effect.buffs['Feather'] / 10
            if 'Fragile' in effect.buffs:
                small_action[25] = effect.buffs['Fragile'] / 10
            if 'Fury' in effect.buffs:
                small_action[6] = effect.buffs['Fury'] / 10
            if 'HealBlock' in effect.buffs:
                small_action[31] = effect.buffs['HealBlock'] / 1.0
            if 'HealingBoost' in effect.buffs:
                small_action[15] = effect.buffs['HealingBoost'] / 10
            if 'Hex' in effect.buffs:
                small_action[32] = effect.buffs['Hex'] / 10

            if 'Leaf' in effect.buffs:
                small_action[16] = effect.buffs['Leaf'] / 10
            if 'Poison' in effect.buffs:
                small_action[34] = effect.buffs['Poison'] / 10
            if 'Rage' in effect.buffs:
                small_action[17] = effect.buffs['Rage'] / 10
            if 'ShieldBoost' in effect.buffs:
                small_action[18] = effect.buffs['ShileBoost'] / 10
            if 'Silence' in effect.buffs:
                small_action[26] = effect.buffs['Silence'] / 10
            if 'Sleep' in effect.buffs:
                small_action[27] = effect.buffs['Sleep'] / 10
            if 'Stealth' in effect.buffs:
                small_action[7] = effect.buffs['Stealth'] / 1.0
            if 'Stunned' in effect.buffs:
                small_action[28] = effect.buffs['Stunned'] / 10
            if 'Taunt' in effect.buffs:
                small_action[19] = effect.buffs['Taunt'] / 10
            if 'Vulnerable' in effect.buffs:
                small_action[29] = effect.buffs['Vulnerable'] / 10
            if 'Weak' in effect.buffs:
                small_action[30] = effect.buffs['Weak'] / 10
            if 'Meditate' in effect.buffs:
                small_action[9] = effect.buffs['Meditate'] / 10

            if 'DeathMark' in effect.buffs:
                small_action[21] = effect.buffs['DeathMark'] / 10
            if 'Bone Sail' in effect.buffs:
                small_action[36] = 1.0
            if 'Cerastes' in effect.buffs:
                small_action[37] = 1.0
            if 'Croc' in effect.buffs:
                small_action[38] = 1.0
            if 'Curved Spine' in effect.buffs:
                small_action[39] = 1.0
            if 'Fish Snack' in effect.buffs:
                small_action[40] = 1.0
            if 'Gravel Ant' in effect.buffs:
                small_action[41] = 1.0
            if 'Hatsune' in effect.buffs:
                small_action[42] = 1.0
            if 'Hermit' in effect.buffs:
                small_action[43] = 1.0
            if 'Indian Star' in effect.buffs:
                small_action[44] = 1.0
            if 'Kingfisher' in effect.buffs:
                small_action[45] = 1.0
            if 'Pumpkin' in effect.buffs:
                small_action[46] = 1.0
            if 'Pupae' in effect.buffs:
                small_action[47] = 1.0
            if 'Scaly Spoon' in effect.buffs:
                small_action[48] = 1.0
            if 'Snail Shell' in effect.buffs:
                small_action[49] = 1.0
            if 'Snake Jar' in effect.buffs:
                small_action[50] = 1.0
            if 'Sponge' in effect.buffs:
                small_action[51] = 1.0
            if 'Teal Shell' in effect.buffs:
                small_action[52] = 1.0
            if 'Timber' in effect.buffs:
                small_action[53] = 1.0
            if 'Watering Can' in effect.buffs:
                small_action[54] = 1.0

            # 召唤物从56 开始编码
            if 'name' in effect.summoner:
                if effect.summoner['name'] == 'Mavis':
                    small_action[56] = 1.0
                elif effect.summoner['name'] == 'Clover':
                    small_action[57] = 1.0
                elif effect.summoner['name'] == 'Little Robin':
                    small_action[58] = 1.0
                elif effect.summoner['name'] == 'Mushroom':
                    small_action[59] = 1.0
                else:
                    print(effect.summoner['name'] + ' is a new summoner ,need to encode')
            return small_action

        def ally_samll_action_encode(effect, small_action):
            if len(effect.damages) != 0:
                small_action[3] = effect.damages[0] / 500
            if len(effect.heals) != 0:
                small_action[4] = effect.heals[0] / 500
            if len(effect.shields) != 0:
                small_action[5] = effect.shields[0] / 500
            if len(effect.hp_changes) != 0:
                small_action[3] = effect.hp_changes[0] / 500

            if 'Bleed' in effect.buffs:
                small_action[20] = effect.buffs['Bleed'] / 10
            if 'Bubble' in effect.buffs:
                small_action[14] = effect.buffs['Bubble'] / 10
            if 'BubbleBomb' in effect.buffs:
                small_action[8] = effect.buffs['BubbleBomb'] / 1.0
            if 'Cleanser' in effect.buffs:
                small_action[11] = effect.buffs['Cleanser'] / 10
            if 'DamageBoost' in effect.buffs:
                small_action[12] = effect.buffs['DamageBoost'] / 10
            if 'Disarmed' in effect.buffs:
                small_action[22] = effect.buffs['Disarmed'] / 10
            if 'Doubt' in effect.buffs:
                small_action[23] = effect.buffs['Doubt'] / 10
            if 'Fear' in effect.buffs:
                small_action[24] = effect.buffs['Fear'] / 10
            if 'Feather' in effect.buffs:
                small_action[13] = effect.buffs['Feather'] / 10
            if 'Fragile' in effect.buffs:
                small_action[25] = effect.buffs['Fragile'] / 10
            if 'Fury' in effect.buffs:
                small_action[6] = effect.buffs['Fury'] / 10
            if 'HealBlock' in effect.buffs:
                small_action[31] = effect.buffs['HealBlock'] / 1.0
            if 'HealingBoost' in effect.buffs:
                small_action[15] = effect.buffs['HealingBoost'] / 10
            if 'Hex' in effect.buffs:
                small_action[32] = effect.buffs['Hex'] / 10

            if 'Leaf' in effect.buffs:
                small_action[16] = effect.buffs['Leaf'] / 10
            if 'Poison' in effect.buffs:
                small_action[34] = effect.buffs['Poison'] / 10
            if 'Rage' in effect.buffs:
                small_action[17] = effect.buffs['Rage'] / 10
            if 'ShieldBoost' in effect.buffs:
                small_action[18] = effect.buffs['ShileBoost'] / 10
            if 'Silence' in effect.buffs:
                small_action[26] = effect.buffs['Silence'] / 10
            if 'Sleep' in effect.buffs:
                small_action[27] = effect.buffs['Sleep'] / 10
            if 'Stealth' in effect.buffs:
                small_action[7] = effect.buffs['Stealth'] / 1.0
            if 'Stunned' in effect.buffs:
                small_action[28] = effect.buffs['Stunned'] / 10
            if 'Taunt' in effect.buffs:
                small_action[19] = effect.buffs['Taunt'] / 10
            if 'Vulnerable' in effect.buffs:
                small_action[29] = effect.buffs['Vulnerable'] / 10
            if 'Weak' in effect.buffs:
                small_action[30] = effect.buffs['Weak'] / 10
            if 'Meditate' in effect.buffs:
                small_action[9] = effect.buffs['Meditate'] / 10

            if 'DeathMark' in effect.buffs:
                small_action[21] = effect.buffs['DeathMark'] / 10

            # 奥秘  只对友方有效 从36开始编码
            if 'Bone Sail' in effect.buffs:
                small_action[36] = 1.0
            if 'Cerastes' in effect.buffs:
                small_action[37] = 1.0
            if 'Croc' in effect.buffs:
                small_action[38] = 1.0
            if 'Curved Spine' in effect.buffs:
                small_action[39] = 1.0
            if 'Fish Snack' in effect.buffs:
                small_action[40] = 1.0
            if 'Gravel Ant' in effect.buffs:
                small_action[41] = 1.0
            if 'Hatsune' in effect.buffs:
                small_action[42] = 1.0
            if 'Hermit' in effect.buffs:
                small_action[43] = 1.0
            if 'Indian Star' in effect.buffs:
                small_action[44] = 1.0
            if 'Kingfisher' in effect.buffs:
                small_action[45] = 1.0
            if 'Pumpkin' in effect.buffs:
                small_action[46] = 1.0
            if 'Pupae' in effect.buffs:
                small_action[47] = 1.0
            if 'Scaly Spoon' in effect.buffs:
                small_action[48] = 1.0
            if 'Snail Shell' in effect.buffs:
                small_action[49] = 1.0
            if 'Snake Jar' in effect.buffs:
                small_action[50] = 1.0
            if 'Sponge' in effect.buffs:
                small_action[51] = 1.0
            if 'Teal Shell' in effect.buffs:
                small_action[52] = 1.0
            if 'Timber' in effect.buffs:
                small_action[53] = 1.0
            if 'Watering Can' in effect.buffs:
                small_action[54] = 1.0

            # 召唤物从56 开始编码
            if 'name' in effect.summoner:
                if effect.summoner['name'] == 'Mavis':
                    small_action[56] = 1.0
                elif effect.summoner['name'] == 'Clover':
                    small_action[57] = 1.0
                elif effect.summoner['name'] == 'Little Robin':
                    small_action[58] = 1.0
                elif effect.summoner['name'] == 'Mushroom':
                    small_action[59] = 1.0
                else:
                    print(effect.summoner['name'] + ' is a new summoner ,need to encode')
            return small_action

        def effect_to_card_action_encode(effect, card):
            effct_to_card_action_encode = np.zeros(30)
            if card.name == 'Beetroot' or 'Blue Moon' or 'Early' or 'Goldfish' or 'Hare' or 'Hero' or 'Tiny':
                if effect.is_initial == True:
                    effct_to_card_action_encode[0] = 1.0

            if card.name == 'Antenna':
                effct_to_card_action_encode[1] = 1.0
            if card.name == 'Blossom':
                effct_to_card_action_encode[2] = 1.0
            if card.name == 'Bookworm':
                effct_to_card_action_encode[3] = 1.0
            if card.name == 'Buzz Buzz':
                effct_to_card_action_encode[4] = 1.0
            if card.name == 'Clear':
                effct_to_card_action_encode[5] = 1.0
            if card.name == 'Confused':
                effct_to_card_action_encode[6] = 1.0
            if card.name == 'Cub':
                effct_to_card_action_encode[7] = 1.0
            if card.name == 'Ear Breathing':
                effct_to_card_action_encode[8] = 1.0
            if card.name == 'Goda':
                effct_to_card_action_encode[9] = 1.0
            if card.name == 'Hot Butt':
                effct_to_card_action_encode[10] = 1.0
            if card.name == 'Inkling':
                effct_to_card_action_encode[11] = 1.0
            if card.name == 'Innocent Lamb':
                effct_to_card_action_encode[12] = 1.0
            if card.name == 'Larva':
                effct_to_card_action_encode[13] = 1.0
            if card.name == 'Lotus':
                effct_to_card_action_encode[14] = 1.0
            if card.name == 'Lucas':
                effct_to_card_action_encode[15] = 1.0
            if card.name == 'Magic Sack':
                effct_to_card_action_encode[16] = 1.0
            if card.name == 'Puppy':
                print('need to change')
            if card.name == 'Raven':
                effct_to_card_action_encode[17] = 1.0
            if card.name == 'Scar':
                effct_to_card_action_encode[18] = 1.0
            if card.name == 'Serious':
                effct_to_card_action_encode[19] = 1.0
            if card.name == 'Sleepless':
                effct_to_card_action_encode[20] = 1.0
            if card.name == 'Swirl':
                effct_to_card_action_encode[21] = 1.0
            if card.name == 'Tadpole':
                effct_to_card_action_encode[22] = 1.0
            if card.name == 'Tricky':
                if effect.is_initial == True:
                    effct_to_card_action_encode[23] = 1.0
            if card.name == 'Unko':
                effct_to_card_action_encode[24] = 1.0
            if card.name == 'Zeal':
                effct_to_card_action_encode[25] = 1.0
            return effct_to_card_action_encode





        if self.sim.battle.current_player is self.players[0]:
            with self.players[0].battle.incognito_mode():
                battle_copy = self.players[0].battle.battle_copy
                cur_player = battle_copy.current_player
                cur_player_hand_card = cur_player.hand_cards
                #card = action_card
                card = None
                for c in cur_player.cards:
                    if c.id == action_card.id:
                        card = c
                        break
                if not card:
                    return
                if card.where != PileType.Hand:
                    # transformed or something else
                    #adprint('error')
                    pass
                if card.has_component("Revenge"):
                    #print(card)
                    pass
                if card.name.startswith("The"):
                    #print(card)
                    pass
                if not card.can_play:
                    #print('error')
                    pass
                if not self.players[0].enemy.alive:
                    # check if enemy is alive (any axie alive)
                    #print('error')
                    pass
                effects = card_effects(battle_copy, card, action_target)
                action_encode = action_encoder(effects, card)


        if self.sim.battle.current_player is self.players[1]:
            with self.players[1].battle.incognito_mode():
                battle_copy = self.players[1].battle.battle_copy
                cur_player = battle_copy.current_player
                cur_player_hand_card = cur_player.hand_cards
                #card = action_card
                card = None
                for c in cur_player.cards:
                    if c.id == action_card.id:
                        card = c
                        break
                if not card:
                    return
                if card.where != PileType.Hand:
                    # transformed or something else
                    print('error_encoder')
                    pass
                if card.has_component("Revenge"):
                    #print(card)
                    pass
                if card.name.startswith("The"):
                    #print(card)
                    pass
                if not card.can_play:
                    #print('error')
                    pass
                if not self.players[1].enemy.alive:
                    # check if enemy is alive (any axie alive)
                    #print('error')
                    pass
                effects = card_effects(battle_copy, card, action_target)
                action_encode = action_encoder(effects, card)
        return action_encode