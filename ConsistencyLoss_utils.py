import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T


class EMA():
    def __init__(self, beta):
        super(EMA, self).__init__()
        self.beta = beta

    def update(self, old, new):
        for current_params, pre_params in zip(new.parameters(), old.parameters()):
            old_weight, new_weight = pre_params.data, current_params.data
            pre_params.data = old_weight * self.beta + new_weight * (1 - self.beta)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, projection_dim):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x):
        return self.layer(x)


class NetWrapper(nn.Module):
    def __init__(self, net, hidden_dim, projection_dim, layer = -2):
        super(NetWrapper, self).__init__()
        self.net = net
        self.layer = layer
        self.projector = None
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.hook_registered = False
        self.hidden = {}

    def _find_layer(self):
        if type(self.layer) == int:
            return [*self.net.children()][self.layer]
        elif type(self.layer) == str:
            modules = dict([*self.net.named_children()])
            return modules.get(self.layer)

    def _hook(self, module, inputs, output):
        bs = output.shape[0]
        device = inputs[0].device
        self.hidden[device] = output.reshape(bs, -1).to(device)


    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, 'Could not find appropriate layer'
        hidden = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def _get_projector(self, representation):
        dim = representation.shape[1]
        projector = MLP(dim, self.hidden_dim, self.projection_dim)
        return projector.to(representation)

    
    def forward(self, x):
        if not self.hook_registered:
            self._register_hook()
        
        logit = self.net(x)
        representation = self.hidden[x.device]
        self.hidden.clear()

        #self.projector = self._get_projector(representation)
        #projection = self.projector(representation)

        return logit, representation

        

