from typing import Sequence

import torch

from .util import build_net


class Actor(torch.nn.Module):
    def __init__(self, max_action: float, net_arch: Sequence[int] = (400, 300)):
        super(Actor, self).__init__()
        self.__net = None
        self.__net_arch = net_arch
        self.__max_action = torch.scalar_tensor(max_action, dtype=torch.float32)

    def initialize(self, state_dim, action_dim):
        self.__net = build_net([state_dim] + list(self.__net_arch) + [action_dim])

    def forward(self, state):
        return self.__max_action * torch.tanh(self.__net(state))
