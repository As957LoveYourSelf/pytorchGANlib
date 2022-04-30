import torch.nn as nn
from base import BaseBackend


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        model = nn.Sequential(

        )

    def forward(self, x):
        pass


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

    def forward(self, x):
        pass


class GD(nn.Module):
    def __init__(self):
        super(GD, self).__init__()

    def forward(self, x):
        pass


class Model(BaseBackend):
    def __init__(self):
        super(Model, self).__init__()
        self._generator = G()
        self._discriminator = D()
        self._deep_network = GD()