import torch
from torch import jit, nn
from torch.nn import functional as F


# TODO: Add a base class Encoder
class SymbolicEncoder(jit.ScriptModule):
    """
    A fully-connected network that learns a mapping from observations to states in the latent space.
    """

    # TODO: According to the paper this should be a recurrent network that takes the last state, the current action, and
    # the current observation into account
    def __init__(self, observation_size: int, embedding_size: int, activation_function: str = "relu"):
        super().__init__()
        # TODO: This is quite hacky
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)

    @jit.script_method
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden
