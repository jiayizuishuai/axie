import os

import torch
import numpy as np

from env_src.envs_axie.axie_feature import Axie_Feature
from env_src.envs_axie.card_feature import Card_Feature
from simulator.base import *
from collections import OrderedDict

class BaseAgent:
    def store(self, data):
        raise NotImplementedError(f"store func must implement")

    def update_infer_model(self):
        raise NotImplementedError(f"update_infer_model func must implement")

    def get_action(self, obs, hidden_state, infer_model_id=0):
        raise NotImplementedError(f"get_action func must implement")

    def get_init_hidden_state(self):
        raise NotImplementedError(f"get_init_hidden_state func must implement")

    def check_update(self):
        raise NotImplementedError(f"check_update func must implement")

    def train(self):
        raise NotImplementedError(f"train func must implement")

    def model_save(self):
        raise NotImplementedError(f"model_save func must implement")


from .models import Model


class DMCV3InferAgent(BaseAgent):#


    #服务端注册的模型
    #store
    #update_infer_model
    #get_action
    def __init__(self, args, data_queue_dict, model_queue_dict,save_model_queue):
        self.args = args
        self.model_names = args.model_names

        self.data_queue_dict = data_queue_dict
        self.model_queue_dict = model_queue_dict
        self.save_model_queue = save_model_queue

        self.config = {'config_path': os.path.join(os.path.abspath(os.path.dirname(__file__)), '../env_src/envs_axie/config')}
        self.axie_feature = Axie_Feature(self.config)
        self.card_feature = Card_Feature(self.config)

        device = args.infer_device

        if not device == "cpu":
            device = 'cuda:' + str(args.train_device)
        self.device = torch.device(device)

        self.infer_models = Model(device=args.infer_device, model_names=self.model_names)
        for model_name in self.model_names:
            #这里将会阻塞住，直到更新参数过来
            parameter = self.model_queue_dict[model_name].get()
            self.update_infer_model(model_name, parameter)

    def store(self, data, model_name):#这是policy_sever调用的函数
        # TODO：是否需要攒一波？
        self.data_queue_dict[model_name].put(data)#将数据保存到队列中
    def save_model(self):
        self.save_model_queue.put(True)

    def update_infer_model(self, model_name, parameter):
        self.infer_models.get_model(model_name).load_state_dict(parameter)

    def get_action(self, position, data, flags):

        if (flags['data_type'] == 'simulator'):
            state = self.axie_feature.get_axie_feature(sim.battle)
            legal_actions, encoded_legal_actions = self.get_legal_actions(sim.battle)

            x_batch = OrderedDict({'state': state,
                                   'legal_actions': legal_actions,
                                   'encoded_legal_actions': encoded_legal_actions})

        elif (flags['data_type'] == 'code'):
            x_batch = data


        agent_output = self.infer_models.forward(position, x_batch, flags=flags)
        _action_idx = int(agent_output['action'].cpu().detach().numpy())



        if (flags['data_type'] == 'simulator'):
            response = {'action': x_batch['legal_actions'][_action_idx],
                        'encoded_action': x_batch['encoded_legal_actions'][_action_idx]}

        elif (flags['data_type'] == 'code'):
            response = {'action_idx': _action_idx,
                        'encoded_action': x_batch['encoded_legal_actions'][_action_idx]}

        # TODO：是否有更优雅的方式检查模型是否更新？
        for model_name in self.model_names:#这里只需要判断  主模型是否更新就可以了
             while not self.model_queue_dict[model_name].empty():
                parameter = self.model_queue_dict[model_name].get()
                self.update_infer_model(model_name, parameter)

        return response

    def get_init_hidden_state(self):
        return None

