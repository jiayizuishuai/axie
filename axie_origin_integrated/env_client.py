import argparse

from core.agent_interact import DMCV3AgentInteract
from env_src.envs_axie.axie import Env_Axie
from simulator.base import TargetType
from simulator.simulator import Simulator
from simulator.entity_builder import loads_every_thing, random_player
from torch import multiprocessing as mp

from agent.dmc_agent import DMCV3Agent

import os
import time
import threading
# loads_every_thing(True)
loads_every_thing()

def env_construct_and_run(i, config, port_id):
    print("actor {} is launching!!".format(i))
    counter = 0
    play0_vic = 0
    counter_sum = 0

    # declare and initialize the environment
    env = Env_Axie(config)
    args = config['args']

    agent_interface = DMCV3AgentInteract(address=f"http://{args.ip}:{port_id}",
                                         positions=config['positions'],
                                         unroll_length=args.unroll_length)

    agent = DMCV3Agent(config['args'])#初始化agent
    while True:
        env.reset()
        if counter > 1000000:
            counter = 0
            play0_vic = 0

        while not env.sim.is_finished():
            env.sim.begin_round()

            # Check whose turn
            current_player_index = None
            if (env.sim.battle.current_player is env.players[0]):
                current_player_index = 0
            else:
                current_player_index = 1

            #obs = env.get_state([env.players[current_player_index],
            #                     env.players[1 - current_player_index],
            #                     env.sim.battle])



            while (not env.sim.is_finished()):
                # TODO using action which is responsed from server
                if (env.sim.battle.current_player is env.players[0]):
                    current_player_index = 0
                    obs_jiayi = env.get_state_jiayi([env.players[0],
                                                     env.players[1],
                                                    env.sim.battle])


                    #thread = threading.Thread(target=action_target.get_action,args=(env.player_ids[current_player_index], obs))

                    action = agent_interface.get_action(position=env.player_ids[current_player_index], obs=obs_jiayi)
                    action_card, action_target = action


                #for test , using local get_action

                    #_action_idx = agent.get_action(env.player_ids[current_player_index], obs_jiayi, flags=None)
                    #action = obs_jiayi['legal_actions'][_action_idx]
                    #action_card, action_target = action


                # For test, using auto play card
                #action_card, action_target = auto_play_card(env.players[current_player_index])

                # break the loop if the action is "end turn"
                    if (action_card == 'end_turn'): break

                # execute this step in env
                    obs_jiayi, reward, done, flags = env.step(action_card=action_card,
                                                    action_target=action_target,
                                                    player_index=current_player_index)

                    if (done): break
                if (env.sim.battle.current_player is env.players[1]):
                    current_player_index = 1
                #    action_card, action_target = auto_play_card(env.players[current_player_index])
                #    if (action_card == 'end_turn'): break
                #    obs, reward, done, flags = env.step(action_card=action_card,
                #                                        action_target=action_target,
                #                                       player_index=current_player_index)
                #    if (action_card == 'end_turn'): break
                    for _ in range(10):
                        env.sim.battle.current_player.random_place_card()
                 #   if (done): break
                    break
            # end this turn
            env.sim.end_round()

        env_output = {}
        for player_index in range(2):
            info = {'episode_return': env.get_reward(player_index)}

            env_output[env.player_ids[player_index]] = info

        agent_interface.end_episode(env_output=env_output)
        counter += 1
        counter_sum+=1
        if env.get_reward(0) == 1:
            play0_vic += env.get_reward(0)
        print("actor_sum {}: ".format(i), counter_sum)
        print("actor {}: ".format(i), counter)
        print('胜率',play0_vic/counter)


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

    parser.add_argument("--port", type=int, default=6789, help="The port to use (on localhost).")

    parser.add_argument("--unroll_length", type=int, default=32, help="the length of data for once sending to server")

    parser.add_argument("--num_actors", type=int, default=1, help='The num of actors for once client launching')

    parser.add_argument("--policy_server_num", type=int, default=1, help='The num of policy servers on server side')


    #下面是服务端的参数


    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        help='Learning rate')

    parser.add_argument('--alpha', default=0.99, type=float,
                        help='RMSProp smoothing constant')

    parser.add_argument('--momentum', default=0, type=float,
                        help='RMSProp momentum')

    parser.add_argument('--epsilon', default=1e-5, type=float,
                        help='RMSProp epsilon')

    parser.add_argument('--max_grad_norm', default=40., type=float,
                        help='Max norm of gradients')

    parser.add_argument('--batch_size', default=100, type=int,
                        help='batch_size')

    parser.add_argument('--buffer_size', default=105, type=int,
                        help='buffer_size')

    parser.add_argument('--save_interval', default=600, type=int,
                        help='The time interval(second) for saving the model parameters')

    parser.add_argument('--model_save_dir', default='./models/', type=str,
                        help='The model save path')

    parser.add_argument('--model_load_dir', default='./models/', type=str,
                        help='The model load path')

    parser.add_argument('--load_model', action='store_true',
                        help='Load an existing model')

    parser.add_argument('--train_device', default='cpu', type=str)  # cpu or 0, 1, 2, ...

    parser.add_argument('--infer_device', default='cpu', type=str)  # cpu or 0, 1, 2, ...

    parser.add_argument('--close_log', default=False, type=bool)


    parser.add_argument('--agent_type', default='train', type=str)

    parser.add_argument('--model_names', default='model_0_player-id_4|model_0_player-id_5', type=str,
       help='The model names')




    args = parser.parse_args()
    card2id_path = os.path.abspath("axie_origin_integrated/env_src/envs_axie/config/card2id.yaml")
    config = {'args': args,
              'positions': ['model_0_player-id_4',
                            'model_0_player-id_5'],
              'path': '/Users/97211/Desktop/axie/site/rct-env-driven-middleware/axie_origin_integrated/env_src/envs_axie/config'}

    env_construct_and_run(0, config,  port_id = args.port + 0 % args.policy_server_num)
    """
    ctx = mp.get_context('spawn')

    for i in range(args.num_actors):
        port_id = args.port + i % args.policy_server_num
        actor = ctx.Process(
            target=env_construct_and_run,
            args=(i, config, port_id))
        actor.start()

    """
