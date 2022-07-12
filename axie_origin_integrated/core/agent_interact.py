import torch as th
import cloudpickle as pickle
import grpc
from rl_grpc.rl_grpc_string import service_pb2_grpc, service_pb2
from config.default_config import GET_ACTION, GET_INIT_HIDDEN_STATE, STORE

import json
import pandas as pd
import os

class BaseAgentInteract:
    def __init__(self, address):
        channel = grpc.insecure_channel(address)
        self.session = service_pb2_grpc.ModelServiceStub(channel)


    def _send(self, data):
        payload = service_pb2.CallRequest(
            input=json.dumps(data),
        )
        response = self.session.Call(payload)
        parsed = json.loads(response.output_specs)
        return parsed

    def start_episode(self):
        raise NotImplementedError(f"start episode func must implement")

    def get_hidden_init(self):
        raise NotImplementedError(f"get_hidden_init func must implement")

    def get_action(self, obs, hidden_state=None):
        raise NotImplementedError(f"get_action func must implement")

    def log_reward_info(self, reward, infos):
        raise NotImplementedError(f"log_reward_info func must implement")

    def end_episode(self, obs):
        raise NotImplementedError(f"end_episode func must implement")

    def store(self):
        raise NotImplementedError(f"store func must implement")


class AgentInteract(BaseAgentInteract):
    def __init__(self, address):
        super().__init__(address)
        # balabala
        self.local_buffer = []

    def start_episode(self):
        self.local_buffer = []
        return True

    def get_hidden_init(self):
        response = self._send(
            {
                "command": GET_INIT_HIDDEN_STATE,
            }
        )
        return response["hidden_state"]

    def get_action(self, obs, hidden_state=None):
        response = self._send(
            {
                "command": GET_ACTION,
                "observation": obs,
                "hidden_state": hidden_state
            }
        )

        self.local_buffer += [obs, response["action"]]
        return response["action"], response["hidden_state"]

    def log_reward_info(self, reward, infos):
        self.local_buffer += [reward, infos]

    def end_episode(self, obs):
        self.local_buffer.append(obs)
        self.store()

    def store(self):
        response = self._send(
            {
                "command": STORE,
                "episode_data": self.local_buffer
            }
        )
        # print(f"send data, response: {response}")

    def _send(self, data):
        payload = pickle.dumps(data)
        response = self.session.post(self.address, data=payload)
        if response.status_code != 200:
            # log
            print(f"Request failed {response.text}: {data}")
        response.raise_for_status()
        parsed = pickle.loads(response.content)
        return parsed


class PPOAgentInteract(AgentInteract):
    def __init__(self, address):
        super().__init__(address)

    def get_action(self, obs, hidden_state=None):
        response = self._send(
            {
                "command": GET_ACTION,
                "observation": obs,
                "hidden_state": hidden_state
            }
        )

        self.local_buffer += [obs, response["action"], response['action_prob']]
        return response["action"], response["hidden_state"]


def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array(list_cards)
    matrix = th.from_numpy(matrix)
    return matrix


class DMCAgentInteract(BaseAgentInteract):
    def __init__(self, address, positions, unroll_length):
        super().__init__(address)
        self.positions = positions
        self.done_buf = {p: [] for p in positions}
        self.episode_return_buf = {p: [] for p in positions}
        self.target_buf = {p: [] for p in positions}
        self.obs_x_no_action_buf = {p: [] for p in positions}
        self.obs_action_buf = {p: [] for p in positions}
        self.obs_z_buf = {p: [] for p in positions}
        self.size = {p: 0 for p in positions}

        self.T = unroll_length

    def start_episode(self):
        return True

    def get_hidden_init(self):
        pass

    def get_action(self, position, obs, env_output, flags):
        self.obs_x_no_action_buf[position].append(
            env_output['obs_x_no_action'])
        self.obs_z_buf[position].append(env_output['obs_z'])

        response = self._send(
            {
                "command": GET_ACTION,
                "position": position,
                "z_batch": obs["z_batch"],
                "x_batch": obs["x_batch"],
                "flags": flags
            }
        )
        # _action_idx = int(agent_output['action'].cpu().detach().numpy())
        _action_idx = response["action"]
        action = obs['legal_actions'][_action_idx]

        self.obs_action_buf[position].append(_cards2tensor(action))
        self.size[position] += 1

        return action

    def log_reward_info(self, reward, infos):
        pass

    def end_episode(self, env_output):
        for p in self.positions:
            diff = self.size[p] - len(self.target_buf[p])
            if diff > 0:
                self.done_buf[p].extend([False for _ in range(diff-1)])
                self.done_buf[p].append(True)

                episode_return = env_output['episode_return'] if p == 'landlord' else - \
                    env_output['episode_return']
                self.episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                self.episode_return_buf[p].append(episode_return)
                self.target_buf[p].extend(
                    [episode_return for _ in range(diff)])

        self.store()

    def store(self):
        for p in self.positions:
            while self.size[p] > self.T:
                local_buffer = {
                    'done': self.done_buf[p][:self.T],
                    'episode_return': self.episode_return_buf[p][:self.T],
                    'target': self.target_buf[p][:self.T],
                    'obs_x_no_action': th.stack(self.obs_x_no_action_buf[p][:self.T], dim=0),
                    'obs_action': th.stack(self.obs_action_buf[p][:self.T], dim=0),
                    'obs_z': th.stack(self.obs_z_buf[p][:self.T], dim=0)
                }

                response = self._send(
                    {
                        "command": STORE,
                        "role": p,
                        "episode_data": local_buffer
                    }
                )

                self.done_buf[p] = self.done_buf[p][self.T:]
                self.episode_return_buf[p] = self.episode_return_buf[p][self.T:]
                self.target_buf[p] = self.target_buf[p][self.T:]
                self.obs_x_no_action_buf[p] = self.obs_x_no_action_buf[p][self.T:]
                self.obs_action_buf[p] = self.obs_action_buf[p][self.T:]
                self.obs_z_buf[p] = self.obs_z_buf[p][self.T:]
                self.size[p] -= self.T

    def _send(self, data):
        payload = pickle.dumps(data)
        response = self.session.post(self.address, data=payload)
        if response.status_code != 200:
            # log
            print(f"Request failed {response.text}: {data}")
        response.raise_for_status()
        parsed = pickle.loads(response.content)
        return parsed


class DMCV3AgentInteract(BaseAgentInteract):
    def __init__(self, address, config_path, unroll_length):
        super().__init__(address)
        self.positions = pd.read_csv(os.path.join(config_path, 'positions_info.csv'))['positions'].to_numpy()
        self.done_buf = {p: [] for p in self.positions}
        self.episode_return_buf = {p: [] for p in self.positions}
        self.target_buf = {p: [] for p in self.positions}
        self.obs_x_no_action_buf = {p: [] for p in self.positions}
        self.obs_action_buf = {p: [] for p in self.positions}
        self.size = {p: 0 for p in self.positions}

        self.T = unroll_length

    def start_episode(self):
        return True

    def get_hidden_init(self):
        pass

    def get_action(self, position, data, flags=None):
        # obs = {
        #   'state': the state of two players and battle infos
        #   'raw_legal_actions' : the list of actions (card_name, card_target)
        #   'legal_actions' : the list of encoded actions (card_name, card_target)
        # }

        response = self._send(
            {
                "command": GET_ACTION,
                "position": position,
                "data": data['encoded_data'],
                "flags": flags
            }
        )

        response = response['response']

        self.obs_x_no_action_buf[position].append(data['encoded_data']['state'])
        self.obs_action_buf[position].append(response['encoded_action'])
        self.size[position] += 1

        if (flags['data_type'] == 'simulator'):
            return response['action']
        elif (flags['data_type'] == 'code'):
            return data['legal_actions'][response['action_idx']]

    def log_reward_info(self, reward, infos):
        pass

    def end_episode(self, env_output=None):
        for p in self.positions:
            diff = self.size[p] - len(self.target_buf[p])
            if diff > 0:
                self.done_buf[p].extend([False for _ in range(diff-1)])
                self.done_buf[p].append(True)

                episode_return = env_output[p]['episode_return']

                self.episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                self.episode_return_buf[p].append(episode_return)
                self.target_buf[p].extend(
                    [episode_return for _ in range(diff)])

        self.store()

    def store(self):
        for p in self.positions:
            while self.size[p] > self.T:

                local_buffer = {
                    'done': self.done_buf[p][:self.T],
                    'episode_return': self.episode_return_buf[p][:self.T],
                    'target': self.target_buf[p][:self.T],
                    'obs_x_no_action': self.obs_x_no_action_buf[p][:self.T],
                    'obs_action': self.obs_action_buf[p][:self.T],
                }

                response = self._send(
                    {
                        "command": STORE,
                        "role": p,
                        "episode_data": local_buffer
                    }
                )

                self.done_buf[p] = self.done_buf[p][self.T:]
                self.episode_return_buf[p] = self.episode_return_buf[p][self.T:]
                self.target_buf[p] = self.target_buf[p][self.T:]
                self.obs_x_no_action_buf[p] = self.obs_x_no_action_buf[p][self.T:]
                self.obs_action_buf[p] = self.obs_action_buf[p][self.T:]
                self.size[p] -= self.T

