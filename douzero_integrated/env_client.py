import argparse

from core.agent_interact import DMCAgentInteract
from douzero_integrated.env_src.env_utils import Environment
from douzero_integrated.env_src.env import Env
from util.draw import terminal_draw
from util.config_r import read_config

def create_env(objective="adp"): # 'adp', 'wp', 'logadp'
    return Env(objective)

if __name__ == "__main__":
    env = create_env()
    env = Environment(env, device='cpu') # or device='0' for gpu 0

    positions = ['landlord', 'landlord_up', 'landlord_down']
    unroll_length = 100

    config_dict = read_config()

    agent_interface = DMCAgentInteract(address=f"http://{config_dict['env_args']['ip']}:{config_dict['env_args']['port']}", positions=positions, unroll_length=unroll_length)

    position, obs, env_output = env.initial()
    episode_return = []

    class Flag:
        def __init__(self, exp_epsilon):
            self.exp_epsilon = exp_epsilon

    while True:
        flags = Flag(config_dict['env_args']['exp_epsilon'])
        action = agent_interface.get_action(position, obs, env_output, flags)

        position, obs, env_output = env.step(action)

        if env_output['done']:
            episode_return.append(env_output['episode_return'][0].cpu().item())
            terminal_draw(episode_return, 'doudizu')
            if env_output['episode_return'] >= config_dict['env_args']['stop_reward']:
                print("Target reward achieved, exiting")
                exit(0)
            
            agent_interface.end_episode(env_output)

