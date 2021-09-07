import torch
from torch import jit, nn
from torch.nn import functional as F


class SymbolicObservationModel(jit.ScriptModule):
    """
    A fully-connected network that learns a mapping from states (in the latent space) to observations for symbolic
    environments. Both the deterministic part ("belief") and the stochastic part ("state") of the recurrent state-space
    model (see PlaNet paper) are fed as input to the neural network.
    """

    def __init__(self, observation_size: int, belief_size: int, state_size: int, embedding_size: int,
                 activation_function: str = "relu"):
        """
        :param observation_size:        the dimensionality of the observation vector
        :param belief_size:             the dimensionality of the deterministic part of the recurrent state-space model
        :param state_size:              the dimensionality of the stochastic part of the recurrent state-space model
        :param embedding_size:
        :param activation_function:     the activation function of nodes in the network
        """
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)

    @jit.script_method
    def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Estimates the observation for a given state (in the recurrent state-space model).

        :param belief:                  the deterministic part of current state
        :param state:                   the stochastic part of current state
        :return:                        the estimated observation
        """
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation
