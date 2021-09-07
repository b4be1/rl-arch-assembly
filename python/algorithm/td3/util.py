from typing import Sequence
import torch.nn as nn


def build_net(arch: Sequence[int]) -> nn.Sequential:
    layers = [(nn.Linear(i, o), nn.ReLU()) for i, o in zip(arch[:-2], arch[1:-1])]
    layers.append((nn.Linear(arch[-2], arch[-1]),))
    return nn.Sequential(*[e for l in layers for e in l])
