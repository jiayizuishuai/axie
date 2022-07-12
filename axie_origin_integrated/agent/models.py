import numpy as np

import torch
from torch import nn


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
            if flags is not None and 'exp_epsilon' in flags.keys() and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0, model_names=[], model_init=True):
        self.models = {}
        self.model_names = model_names

        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.device = device

        if (model_init):
            for model_name in model_names:
                self.models[model_name] = AxieModel().to(torch.device(device))


    def forward(self, position, x, training=False, flags=None):
        states = np.repeat([x['state']], len(x['encoded_legal_actions']), axis=0)
        actions = x['encoded_legal_actions']

        x = np.concatenate((states, actions), 1)
        x = torch.from_numpy(x).to(self.device)

        model = self.models[position]
        return model.forward(x, training, flags)

    def share_memory(self):
        for model_name in self.model_names:
            self.models[model_name].share_memory()

    def eval(self):
        for model_name in self.model_names:
            self.models[model_name].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models

    def add_model(self, model, model_name):
        self.model_names.append(model_name)
        self.models[model_name] = model

