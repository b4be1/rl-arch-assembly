from typing import Union

import torch


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6),
                 training_device: Union[str, torch.device] = "cpu", storage_device: Union[str, torch.device] = "cpu"):
        self.__max_size = max_size
        self.__ptr = 0
        self.__size = 0

        self.__training_device = training_device if isinstance(training_device, torch.device) else torch.device(
            training_device)
        self.__storage_device = storage_device if isinstance(storage_device, torch.device) else torch.device(
            storage_device)

        self.__state = torch.zeros((max_size, state_dim), device=self.__storage_device)
        self.__action = torch.zeros((max_size, action_dim), device=self.__storage_device)
        self.__next_state = torch.zeros((max_size, state_dim), device=self.__storage_device)
        self.__reward = torch.zeros((max_size, 1), device=self.__storage_device)
        self.__not_done = torch.zeros((max_size, 1), device=self.__storage_device)

    def add(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor,
            done: torch.Tensor):
        self.__state[self.__ptr] = state.to(self.__storage_device)
        self.__action[self.__ptr] = action.to(self.__storage_device)
        self.__next_state[self.__ptr] = next_state.to(self.__storage_device)
        self.__reward[self.__ptr] = reward.to(self.__storage_device)
        self.__not_done[self.__ptr] = torch.logical_not(done).to(self.__storage_device)

        self.__ptr = (self.__ptr + 1) % self.__max_size
        self.__size = min(self.__size + 1, self.__max_size)

    def sample(self, batch_size: int):
        idx = torch.randint(low=0, high=self.__size, device=self.__storage_device, size=(batch_size, ))

        return (
            torch.FloatTensor(self.__state[idx]).to(self.__training_device),
            torch.FloatTensor(self.__action[idx]).to(self.__training_device),
            torch.FloatTensor(self.__next_state[idx]).to(self.__training_device),
            torch.FloatTensor(self.__reward[idx]).to(self.__training_device),
            torch.FloatTensor(self.__not_done[idx]).to(self.__training_device)
        )

    @property
    def size(self) -> int:
        return self.__size
