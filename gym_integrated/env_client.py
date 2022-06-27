import argparse

import gym

from core.agent_interact import AgentInteract, PPOAgentInteract
from util.draw import terminal_draw
from util.config_r import read_config

if __name__ == "__main__":
    config_dict = read_config()

    env = gym.make(config_dict['env_args']['env_name'])

    if config_dict['env_args']['agent_interact_type'] == "ppo":
        agent_interface = PPOAgentInteract(
            address=f"http://{config_dict['env_args']['ip']}:{config_dict['env_args']['port']}")
    else:
        agent_interface = AgentInteract(
            address=f"http://{config_dict['env_args']['ip']}:{config_dict['env_args']['port']}")

    obs = env.reset()
    agent_interface.start_episode()
    hidden = agent_interface.get_hidden_init()
    rewards = 0.0
    total_reward_list = []
    while True:
        action, hidden = agent_interface.get_action(obs, hidden)

        obs, reward, done, info = env.step(action)
        rewards += reward

        agent_interface.log_reward_info(reward, info)

        if done:
            total_reward_list.append(rewards)
            terminal_draw(total_reward_list, config_dict['env_args']['env_name'])
            if rewards >= config_dict['env_args']['stop_reward']:
                print("Target reward achieved, exiting")
                exit(0)

            rewards = 0.0

            agent_interface.end_episode(obs)

            obs = env.reset()
            agent_interface.start_episode()
