from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Tuple

import numpy as np

if TYPE_CHECKING:
    from algorithm.algorithm import Algorithm


class Callback:
    def __init__(self):
        self.__algorithm = None
        self.__total_timesteps = None

    def init_training(self, algorithm: "Algorithm", total_timesteps: int):
        self.__algorithm = algorithm
        self.__total_timesteps = total_timesteps

    def on_step(self, current_timestep: int, action: Tuple[np.ndarray], obs: Tuple[np.ndarray], reward: Tuple[float],
                done: Tuple[float], info: Tuple[Dict[str, Any]]):
        pass

    def on_episode_end(self):
        pass

    def on_episode_start(self, obs: Tuple[np.ndarray]):
        pass

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    @property
    def algorithm(self) -> "Algorithm":
        return self.__algorithm

    @property
    def total_timesteps(self) -> int:
        return self.__total_timesteps
