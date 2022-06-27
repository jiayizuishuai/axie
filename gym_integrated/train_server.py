import argparse
import threading

import gym

from agent.a2c.a2c_agent import A2CAgent
from agent.dqn.dqn_agent import DQNAgent
from agent.ppo.ppo_agent import PPOAgent
from core.policy_server import PolicyServer
from core.base_agent import train
from util.config_r import read_config

if __name__ == "__main__":
    config_dict = read_config()
    print(config_dict)

    local_env = gym.make(config_dict['env_args']['env_name'])
    observation_space = local_env.observation_space
    action_space = local_env.action_space

    if config_dict['algorithm'] == 'a2c':
        agent = A2CAgent(
            args=config_dict['algo_args'], observation_space=observation_space, action_space=action_space)
    elif config_dict['algorithm'] == 'dqn':
        agent = DQNAgent(
        args=config_dict['algo_args'], observation_space=observation_space, action_space=action_space)
    elif config_dict['algorithm'] == 'ppo':
        agent = PPOAgent(
        args=config_dict['algo_args'], observation_space=observation_space, action_space=action_space)
    else:
        raise ValueError()

    train_thread = threading.Thread(
        target=train, name='learn', kwargs={'agent': agent})
    train_thread.start()

    if config_dict['algorithm'] == 'ppo':
        policy_server = PolicyServer(
            agent=agent, address='0.0.0.0', port=config_dict['algo_args']['port'], handler_type='ppo')
    else:
        policy_server = PolicyServer(
            agent=agent, address='0.0.0.0', port=config_dict['algo_args']['port'])
    policy_server.serve_forever()
