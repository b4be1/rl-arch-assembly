from typing import Sequence, Tuple

import torch
from torch.nn import ModuleList

from .util import build_net


class Critic(torch.nn.Module):
    def __init__(self, net_arch: Sequence[int] = (400, 300), num_q_nets: int = 2):
        super(Critic, self).__init__()

        self.__num_q_nets = num_q_nets
        self.__net_arch = net_arch

        # Q1/Q2 architecture
        self.__q_nets = None

    def initialize(self, state_dim: int, action_dim: int):
        self.__q_nets = ModuleList(
            build_net([state_dim + action_dim] + list(self.__net_arch) + [1]) for _ in range(self.__num_q_nets))

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return tuple(q(sa) for q in self.__q_nets)

    @property
    def q_nets(self) -> Tuple[torch.nn.Module, ...]:
        return tuple(self.__q_nets)
