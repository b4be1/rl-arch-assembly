from typing import Tuple

import numpy as np
import torch
from .env import postprocess_observation, preprocess_observation_


class ExperienceReplay:
    """
    A replay buffer that stores (observation, action, reward, not done) tuples. The replay buffer can be queried to
    return batches of chunks. A chunk is a sequence of consecutive (observation, action, reward, not done) tuples.
    If the buffer is full, old data will be overwritten (it is essentially a ring buffer).
    """
    # TODO: Chunks seem to potentially contain tuples from different episodes (which are thus not consecutive)
    # TODO: Why is done negated?

    def __init__(self, size: int, symbolic_env: bool, observation_size: int, action_size: int, bit_depth: int = None,
                 device: torch.device = torch.device("cpu")):
        """
        Creates a replay buffer.

        :param size:                the size (number of tuples stored) of the replay buffer
        :param symbolic_env:        whether the gym env is symbolic (i.e. the observations are not images)
        :param observation_size:    the length of the observation vector
        :param action_size:         the length of the action vector
        :param bit_depth:           the bit depth to use for storing images (ignored for symbolic environments)
                                    --> images are discretized to [0, 2 ** bit_depth]
        :param device:              the torch device (torch.device("cpu") or torch.device("cuda"))
        """
        self.device = device
        self.symbolic_env = symbolic_env
        self.size = size
        self.observations: np.ndarray = np.empty((size, observation_size) if symbolic_env else (size, 3, 64, 64),
                                        dtype=np.float32 if symbolic_env else np.uint8)
        self.actions: np.ndarray = np.empty((size, action_size), dtype=np.float32)
        self.rewards: np.ndarray = np.empty((size,), dtype=np.float32)
        self.nonterminals: np.ndarray = np.empty((size, 1), dtype=np.float32)
        self.idx: int = 0           # The index of the next free element in the replay buffer
        self.full: bool = False     # Tracks if memory has been filled/all slots are valid
        # TODO: The next two attributes dont really belong into this class (and are not needed)
        self.steps: int = 0         # Tracks the total number of steps that were added to the replay buffer
        self.episodes: int = 0      # Tracks the total number of episodes that were added to the replay buffer
        self.bit_depth = bit_depth

    def append(self, observation: torch.Tensor, action: torch.Tensor, reward: float, done: bool) -> None:
        """
        Adds a tuple to the replay buffer.

        :param observation:     the observation at the current time step
        :param action:          the action at the current time step
        :param reward:          the reward at the current time step
        :param done:            the done signal at the current time step
        """
        if self.symbolic_env:
            self.observations[self.idx] = observation
        else:
            # Decentre and discretize visual observations (to save memory)
            self.observations[self.idx] = postprocess_observation(observation, self.bit_depth)
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps = self.steps + 1
        self.episodes = self.episodes + (1 if done else 0)

    def _sample_idx(self, chunk_size: int) -> np.ndarray:
        """
        Returns an array of the indices of a uniformly sampled chunk from the memory.

        :param chunk_size:          the chunk size
        :return:                    an array of the indices of a random chunk
        """
        # To make PyLint shut up
        idxs = None
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - chunk_size + 1)
            idxs = np.arange(idx, idx + chunk_size) % self.size
            # TODO: Shouldnt that already be ensured by the previous line
            valid_idx = self.idx not in idxs[1:]  # Make sure data does not cross the memory index
        return idxs

    # TODO: Use either only tensors or only array
    def _retrieve_batch(self, idxs: np.ndarray) -> Tuple[torch.tensor, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves a batch of chunks from the replay buffer.

        :param idxs:        the indices of the tuples belonging to the batch as a matrix (idxs[chunk_idx, sample_idx] is
                            the index in the replay buffer that corresponds to the sample with index sample_idx in the
                            chunk with index chunk_idx)
        :return:            a tuple (observations, actions, rewards, not dones) where each entry is a
                            (chunk_size, batch_size, ...) array / tensor
        """
        batch_size, chunk_size = idxs.shape
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
        if not self.symbolic_env:
            preprocess_observation_(observations, self.bit_depth)  # Undo discretization for visual observations
        # TODO: nonterminals should be a 2D array
        # TODO: Why is now the first index the sample index and the second the chunk index?
        return observations.reshape(chunk_size, batch_size, *observations.shape[1:]), \
               self.actions[vec_idxs].reshape(chunk_size, batch_size, -1), \
               self.rewards[vec_idxs].reshape(chunk_size, batch_size), \
               self.nonterminals[vec_idxs].reshape(chunk_size, batch_size, 1)

    def sample(self, batch_size: int, chunk_size: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a batch of sequence chunks uniformly sampled from the memory.

        :param batch_size:      the size (nr of chunks) of the batch
        :param chunk_size:      the size of the chunks
        :return:                a tuple (observations, actions, rewards, not dones) where each entry is a
                                (chunk_size, batch_size, ...) tensor
        """
        batch = self._retrieve_batch(
            np.asarray([self._sample_idx(chunk_size) for _ in range(batch_size)]))
        return tuple(torch.as_tensor(item).to(device=self.device) for item in batch)
