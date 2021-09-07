import torch
from torch import jit, nn
from torch.nn import functional as F


class VisualObservationModel(jit.ScriptModule):
    __constants__ = ["embedding_size"]

    def __init__(self, belief_size: int, state_size: int, embedding_size: int, activation_function: str = "relu"):
        super().__init__()
        # TODO: This is quite hacky
        self.act_fn = getattr(F, activation_function)
        self.hidden_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    @jit.script_method
    def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.hidden_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation
