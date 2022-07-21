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

from agent.dmc_infer_agent import DMCV3InferAgent
from core.policy_server import PolicyServer
import pandas as pd
from multiprocessing import Process, Queue

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
    evaluate = True
    if evaluate  == False:
        win = 0
        for i in range(128):
            env.reset()
            while not env.sim.is_finished():
                env.sim.begin_round()

                current_player_index = None
                if (env.sim.battle.current_player is env.players[0]):
                    current_player_index = 0
                else:
                    current_player_index = 1
                while (not env.sim.is_finished()):
                    data = env.get_state(list_type=True),

                    action_id = agent.get_action(position=env.player_ids[current_player_index],
                                              data=data,
                                              flags={'data_type': 'code'})['action_idx']

                    action_card, action_target = data[0]['legal_actions'][action_id]

                    # break the loop if the action is "end turn"
                    if (action_card == 'end_turn'): break

                    # execute this step in env
                    _, _, done, _ = env.step(action_card=action_card,
                                             action_target=action_target,
                                             player_index=current_player_index)

                    if (done): break

                    # end this turn
                env.sim.end_round()


            info = {'episode_return': env.get_reward(0)}
            if info['episode_return']  == 1:
                win += 1
        win = win/128
        print('胜率为：'+str(win))




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
                data = env.get_state(list_type=True)

                action = agent.get_action(position=env.player_ids[current_player_index],
                                                    data=data,
                                                    flags={'data_type': 'code'})



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

    parser.add_argument("--num_actors", type=int, default=1, help='The num of actors for once client launching')

    parser.add_argument("--policy_server_num", type=int, default=1, help='The num of policy servers on server side')


    #服务端参数


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

    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch_size')

    parser.add_argument('--buffer_size', default=1050, type=int,
                        help='buffer_size')

    parser.add_argument('--save_interval', default=600, type=int,
                        help='The time interval(second) for saving the model parameters')

    parser.add_argument('--log_interval', default=10, type=int,
                        help='The time interval(second) for log the train agent stats')

    parser.add_argument('--model_save_dir', default='./models/', type=str,
                        help='The model save path')

    parser.add_argument('--model_load_dir', default='./models/', type=str,
                        help='The model load path')

    parser.add_argument('--load_model_server', default='True',
                        help='Load an existing model')

    parser.add_argument('--load_model_env', default='True',
                        help='Load an existing model')


    parser.add_argument('--train_device', default='cpu', type=str)  # cpu or 0, 1, 2, ...

    parser.add_argument('--infer_device', default='cpu', type=str)  # cpu or 0, 1, 2, ...

    parser.add_argument('--close_log', default=False, type=bool)



    parser.add_argument('--agent_type', default='train', type=str,
                        help='train: this agent is used as training agent, can infering and training; '
                             'service: this agent has no training service')






    args = parser.parse_args()
    agent = None

    def create_infer(args, data_queue_dict, model_queue_dict, port):
        agent = DMCV3InferAgent(args=args,
                                data_queue_dict=data_queue_dict,
                                model_queue_dict=model_queue_dict)




    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'env_src/envs_axie/config')
    model_names = pd.read_csv(os.path.join(config_path, 'positions_info.csv'))['positions'].values.tolist()

    args.model_names = model_names
    num_parallel = args.policy_server_num
    data_queue_dict_list = [{model_name: Queue() for model_name in model_names} for _ in range(num_parallel)]
    model_queue_dict_list = [{model_name: Queue() for model_name in model_names} for _ in range(num_parallel)]

    for i, (data_queue_dict, model_queue_dict) in enumerate(zip(data_queue_dict_list, model_queue_dict_list)):
        agent = DMCV3InferAgent(args,data_queue_dict,model_queue_dict)

    agent_env = DMCV3Agent(args, data_queue_dict_list, model_queue_dict_list,env = True)



    config = {'args': args,
              'config_path': os.path.join(os.path.abspath(os.path.dirname(__file__)), 'env_src/envs_axie/config')}

    env_construct_and_run(0, config, 5789)

    ctx = mp.get_context('spawn')

    for i in range(args.num_actors):
        port_id = args.port + i % args.policy_server_num
        actor = ctx.Process(
            target=env_construct_and_run,
            args=(i, config, port_id))
        actor.start()





