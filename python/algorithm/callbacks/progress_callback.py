import logging
from typing import Dict, Any, Tuple

import numpy as np

from algorithm.callbacks.callback import Callback


class ProgressCallback(Callback):
    def __init__(self, print_interval: int, logger: logging.Logger):
        super(ProgressCallback, self).__init__()
        self.__print_interval = print_interval
        self.__logger = logger
        self.__last_log = -print_interval

    def on_step(self, current_timestep: int, action: Tuple[np.ndarray], obs: Tuple[np.ndarray], reward: Tuple[float],
                done: Tuple[float], info: Tuple[Dict[str, Any]]):
        if current_timestep - self.__last_log >= self.__print_interval:
            self.__logger.log(logging.INFO, "Progress: {}/{} timesteps".format(current_timestep, self.total_timesteps))
            self.__last_log += self.__print_interval
