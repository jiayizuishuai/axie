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
from agent.dmc_env_forward_agent import DMCV3Agent_forward
import time


loads_every_thing()





def evaluate( main_model,evaluate_model_name):
    '''

    :param env:   将环境输入进来
    :param evaluate_model_name: 选择要评估的模型
    :return:
    嘉义的evaluate功能，
    '''
    agent_interface = main_model
    forward_agent = evaluate_model_name
    env = Env_Axie(config)
    win = 0
    for i in range(100):
        env.reset()
        while not env.sim.is_finished():
            env.sim.begin_round()

            current_player_index = None
            if (env.sim.battle.current_player is env.players[0]):
                current_player_index = 0
            else:
                current_player_index = 1
            while (not env.sim.is_finished()):

                if current_player_index == 0 :
                    action = agent_interface.get_action_evaluate(position=env.player_ids[current_player_index],
                                                        data=env.get_state(list_type=True),
                                                        flags={'data_type': 'code'})

                if current_player_index == 1 :
                    action = forward_agent.get_action(data = env.get_state(list_type=True),flags={'data_type':'code'})





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


        info = {'episode_return': env.get_reward(0)}
        if info['episode_return']  == 1:
            win += 1
    win = win/128
    print('胜率为：'+str(win))
    return win





def env_construct_and_run(i, config, port_id,forward_agent):
    # declare and initialize the environment
    forward_agent = forward_agent
    env = Env_Axie(config)
    args = config['args']
    #主模型与服务端的交互接口
    agent_interface = DMCV3AgentInteract(address=f"{args.ip}:{port_id}",
                                         config_path=config['config_path'],
                                         unroll_length=args.unroll_length)
    game_nums = 0
    while game_nums < 300:
        #开始新的一局
        game_nums += 1
        print('当前子进程运行了'+str(game_nums)+'局')
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


                if current_player_index == 0:

                    action = agent_interface.get_action(position=env.player_ids[current_player_index],
                                                        data=env.get_state(list_type=True),
                                                        flags={'data_type':'code'})

                if current_player_index == 1:
                    action = forward_agent.get_action(data = env.get_state(list_type=True),flags={'data_type':'code'})



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






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", type=str, default="127.0.0.1", help="The ip to use.")

    parser.add_argument("--port", type=int, default=5789, help="The port to use (on localhost).")

    parser.add_argument("--unroll_length", type=int, default=32, help="the length of data for once sending to server")

    parser.add_argument("--num_actors", type=int, default=20, help='The num of actors for once client launching')

    parser.add_argument("--policy_server_num", type=int, default=1, help='The num of policy servers on server side')


    args = parser.parse_args()


    config = {'args': args,
              'config_path': os.path.join(os.path.abspath(os.path.dirname(__file__)), 'env_src/envs_axie/config')}

    #env_construct_and_run(0, config, 5789)
    test_nums = 0

    ctx = mp.get_context('spawn')
    '''
    for i in range(args.num_actors):
        port_id = args.port + i % args.policy_server_num
        actor = ctx.Process(
            target=env_construct_and_run,
            args=(i, config, port_id))
        actor.start()

    '''


    #主模型永远一个
    agent_interface = DMCV3AgentInteract(address=f"{args.ip}:{5789}",
                                         config_path=config['config_path'],
                                         unroll_length=args.unroll_length)
    '''
    #用于主模型保存模型的判断
    model_save_interface = DMCV3_save_model_interact(address=f"{args.ip}:{7788}",
                                         config_path=config['config_path'],
                                         unroll_length=args.unroll_length)
    '''
    while True:
        #从0开始学
        model_history_path = './models/model_history.txt'
        file = open(model_history_path, 'r')
        model_history = file.read()
        file.close()
        forward_agent_name = 'model_' +str(model_history)+ '_player-id_52'
        forward_agent= DMCV3Agent_forward(forward_agent_name)
        #读取前馈模型作为对手
        for i in range(args.num_actors):
            port_id = args.port + i % args.policy_server_num
            actor = ctx.Process(
                target=env_construct_and_run,
                args=(i, config, port_id,forward_agent))
            actor.start()

            #每个子进程都结束，继续主进程
        actor.join()
        print('所有子进程都结束了')
        #验证模型效果,每个子进程跑300局验证一次效果
        #time.sleep(1000)
        test_win = evaluate(main_model = agent_interface ,evaluate_model_name= forward_agent)
        if test_win > 0.6:
            #1.保存模型

            agent_interface.save_model()
            #2.开始新的前馈模型
            model_history_path = './models/model_history.txt'
            file = open(model_history_path, 'r')
            model_history = file.read()
            file.close()
            forward_agent_name = 'model_' + str(model_history) + '_player-id_52'
            forward_agent = DMCV3Agent_forward(forward_agent_name)
            # 读取前馈模型作为对手
            for i in range(args.num_actors):
                port_id = args.port + i % args.policy_server_num
                actor = ctx.Process(
                    target=env_construct_and_run,
                    args=(i, config, port_id, forward_agent))
                actor.start()
            actor.join()


        if test_win < 0.6 :
            print('小于0.6继续学习，不保存模型胜率为'+str(test_win))
            pass

