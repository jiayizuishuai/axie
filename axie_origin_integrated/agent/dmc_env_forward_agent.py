'''
嘉义
只加载一个模型永远前馈
'''
import os
import torch
import numpy as np
from torch import nn
import numpy
class AxieModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(1264 + 241, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, x, return_value=False, flags=None):
        x = x.float()
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)

        if return_value:
            return dict(values=x)
        else:
            if flags is not None and 'exp_epsilon' in flags.keys() and flags['exp_epsilon']> 0 and np.random.rand() < flags['exp_epsilon']:

                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device, model_init=True):
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.device = device
        if (model_init):
            self.model = AxieModel().to(torch.device(device))


    def forward(self, x, training=False, flags=None):
        states = np.repeat([x['state']], len(x['encoded_legal_actions']), axis=0)
        actions = x['encoded_legal_actions']
        x = np.concatenate((states, actions), 1)
        x = torch.from_numpy(x).to(self.device)
        return self.model.forward(x, training, flags)






class DMCV3Agent_forward():
    def __init__(self,model_name):
        self.model_name = model_name
        self.forward_model = Model(device='cpu')  # 初始化模型
        if model_name == 'model_-1_player-id_52':
            print('加载了一个随机模型')
        else:
            #客户端加载了模型，但是不检查更新永远前馈，也不会存储数据
            env_model =os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/models/' + model_name + '/'
            checkpoint_states = torch.load(env_model + "model.tar", map_location=('cpu'))
            self.forward_model.model.load_state_dict(checkpoint_states['model_state_dict'])

    def get_action(self,data,flags):
        x_batch = data['encoded_data']
        agent_output = self.forward_model.forward( x_batch, flags=flags)
        _action_idx = int(agent_output['action'].cpu().detach().numpy())
        return data['legal_actions'][_action_idx]



if __name__ == '__main__':
    random = np.random.rand(1)
    model_history = 0
    if random < 0.3:
        model_history = 'main'
    else :

        model_history_path=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/models/model_history.txt'

        file = open(model_history_path, 'r')
        model_history = file.read()
        file.close()
        model_history = int(model_history)

        #选择模型
        env_model_name = 'model_'+str(model_history)+'_player-id_52'



    agent = DMCV3Agent_forward('model_0_player-id_52')

