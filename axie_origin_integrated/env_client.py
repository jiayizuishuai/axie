import argparse

from core.agent_interact import DMCV3AgentInteract
from axie_origin_integrated.env_src.envs_axie.axie import Env_Axie
from simulator.base import TargetType
from simulator.simulator import Simulator
from simulator.entity_builder import loads_every_thing, random_player
from torch import multiprocessing as mp


import time

# loads_every_thing(True)
loads_every_thing()

def env_construct_and_run(i, config, port_id):
    print("actor {} is launching!!".format(i))
    counter = 0

    # declare and initialize the environment
    env = Env_Axie(config)
    args = config['args']

    agent_interface = DMCV3AgentInteract(address=f"http://{args.ip}:{port_id}",
                                         positions=config['positions'],
                                         unroll_length=args.unroll_length)

    while True:
        env.reset()

        while not env.sim.is_finished():
            env.sim.begin_round()

            # Check whose turn
            current_player_index = None
            if (env.sim.battle.current_player is env.players[0]):
                current_player_index = 0
            else:
                current_player_index = 1

            obs = env.get_state([env.players[current_player_index],
                                 env.players[1 - current_player_index],
                                 env.sim.battle])

            while (not env.sim.is_finished()):
                action = agent_interface.get_action(position=env.player_ids[current_player_index], obs=obs)
                action_card, action_target = action

                # break the loop if the action is "end turn"
                if (action_card == 'end_turn'): break

                # execute this step in env
                obs, reward, done, flags = env.step(action_card=action_card,
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
        counter += 1
        print("actor {}: ".format(i), counter)



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

    args = parser.parse_args()

    config = {'args': args,
              'positions': ['model:0_player-id:4',
                            'model:0_player-id:5'],
              'path': '/Users/francis/Public/francis/project/rct-env-driven-middleware/axie_origin_integrated/env_src/envs_axie/config'}

    ctx = mp.get_context('spawn')

    for i in range(args.num_actors):
        port_id = args.port + i % args.policy_server_num
        actor = ctx.Process(
            target=env_construct_and_run,
            args=(i, config, port_id))
        actor.start()


