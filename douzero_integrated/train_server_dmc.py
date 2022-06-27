import argparse
from ast import arg
import threading

from douzero_integrated.agent.dmc_agent import DMCAgent
from core.policy_server import PolicyServer
from core.base_agent import train
from util.config_r import read_config


if __name__ == "__main__":
    config_dict = read_config()
    print(config_dict)

    agent = DMCAgent(args=config_dict['algo_args'])

    train_thread = threading.Thread(
        target=train, name='learn', kwargs={'agent': agent})
    train_thread.start()

    policy_server = PolicyServer(
        agent=agent, address='0.0.0.0', port=config_dict['algo_args']['port'], handler_type='dmc', close_log=config_dict['algo_args']['close_log'])
    policy_server.serve_forever()
