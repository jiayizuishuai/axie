import argparse
import os

from core.agent_interact import DMCV3AgentInteract
from env_src.envs_axie.axie import Env_Axie
from simulator.base import TargetType
from simulator.simulator import Simulator
from simulator.entity_builder import loads_every_thing, random_player
from torch import multiprocessing as mp
from agent.dmc_agent import DMCV3Agent
import torch

import time

#loads_every_thing(True)
loads_every_thing()

def env_construct_and_run(i, config, port_id):
    # declare and initialize the environment
    env = Env_Axie(config)
    args = config['args']

    agent_interface = DMCV3AgentInteract(address=f"{args.ip}:{port_id}",
                                         config_path=config['config_path'],
                                         unroll_length=args.unroll_length)




    while True:
        env.reset()

        while not env.sim.is_finished():
            round_begin_time = time.time()

            env.sim.begin_round()

            # Check whose turn
            current_player_index = None
            if (env.sim.battle.current_player is env.players[0]):
                current_player_index = 0
            else:
                current_player_index = 1

            while (not env.sim.is_finished()):
                state = env.axie_feature.get_axie_feature(env.sim.battle)





                # pass JSON verions
                action = agent_interface.get_action(position=env.player_ids[current_player_index],
                                                    data=env.get_state(list_type=True),
                                                    flags={'data_type': 'code'})

                action_card, action_target = action

                # break the loop if the action is "end turn"
                if (action_card == 'end_turn'): break

                # execute this step in env
                _, _, done, _ = env.step(action_card=action_card,
                                         action_target=action_target,
                                         player_index=current_player_index)

                if (done): break


            # end this turn
            env.sim.end_round()

        env_output = {}
        for player_index in range(2):
            info = {'episode_return': env.get_reward(player_index)}

            env_output[env.player_ids[player_index]] = info

        agent_interface.end_episode(env_output=env_output)



def auto_play_card(player):
    cards = sorted(player.hand_cards, key=lambda x: x.damage, reverse=True)

    if (player.energy == 0):
        return "end_turn", -1

    for card in cards:
        if not card.can_play:
            continue

        if not player.enemy.alive:
            # check if enemy is alive (any axie alive)
            break

        target_type = card.target_type
        if target_type == TargetType.Auto:
            return card, -1

        elif target_type == TargetType.Ally:
            for idx, axie in enumerate(player.positions):
                if axie:
                    return card, idx

        elif target_type == TargetType.Enemy:
            for idx, axie in enumerate(player.enemy.positions):
                if axie:
                    return card, idx

    return "end_turn", -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", type=str, default="127.0.0.1", help="The ip to use.")

    parser.add_argument("--port", type=int, default=5789, help="The port to use (on localhost).")

    parser.add_argument("--unroll_length", type=int, default=32, help="the length of data for once sending to server")

    parser.add_argument("--num_actors", type=int, default=50, help='The num of actors for once client launching')

    parser.add_argument("--policy_server_num", type=int, default=10, help='The num of policy servers on server side')

    args = parser.parse_args()

    config = {'args': args,
              'config_path': os.path.join(os.path.abspath(os.path.dirname(__file__)), 'env_src/envs_axie/config')}




    ctx = mp.get_context('spawn')

    for i in range(args.num_actors):
        port_id = args.port + i % args.policy_server_num
        actor = ctx.Process(
            target=env_construct_and_run,
            args=(i, config, port_id))
        actor.start()





