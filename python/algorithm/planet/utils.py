from typing import Callable, Iterable

import torch


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features
# (assumes one output)
# TODO: Make Callable a bit more precise
def bottle(f: Callable, x_tuple: Iterable[torch.Tensor]) -> torch.Tensor:
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])

