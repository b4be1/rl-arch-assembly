import torch
from torch import jit, nn
from torch.nn import functional as F


class RewardModel(jit.ScriptModule):
    """
    A fully-connected network that learns a mapping from states (in the latent space) to rewards. Both the deterministic
    part ("belief") and the stochastic part ("state") of the recurrent state-space model (see PlaNet paper) are fed as
    input to the neural network.
    """

    # TODO: Should also take actions into account.
    def __init__(self, belief_size: int, state_size: int, hidden_size: int, activation_function: str = "relu"):
        """
        :param belief_size:             the dimensionality of the deterministic part of the recurrent state-space model
        :param state_size:              the dimensionality of the stochastic part of the recurrent state-space model
        :param hidden_size:             the number of neurons of the hidden layer in the network
        :param activation_function:     the activation function of nodes in the network
        """
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    @jit.script_method
    def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Estimates the reward for a given state (in the recurrent state-space model).

        :param belief:                  the deterministic part of current state
        :param state:                   the stochastic part of current state
        :return:                        the estimated reward
        """
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward
