from pathlib import Path

from typing import Dict, Any, Tuple

import numpy as np

from algorithm.callbacks.callback import Callback


class SaveModelCallback(Callback):
    def __init__(self, interval: int, base_path: Path):
        super(SaveModelCallback, self).__init__()
        self.__interval = interval
        self.__base_path = base_path
        self.__models_path = base_path / "models"
        self.__models_path.mkdir()
        self.__last_save = -interval

    def on_step(self, current_timestep: int, action: Tuple[np.ndarray], obs: Tuple[np.ndarray], reward: Tuple[float],
                done: Tuple[float], info: Tuple[Dict[str, Any]]):
        if current_timestep - self.__last_save >= self.__interval:
            digit_count = int(np.floor(np.log10(self.total_timesteps))) + 1
            save_path = self.__models_path / "model_{:0{}d}.pkl".format(current_timestep, digit_count)
            link_path = self.__base_path / "model.pkl"
            link_path.unlink(missing_ok=True)
            link_path.symlink_to(save_path.relative_to(self.__base_path))
            self.algorithm.save_checkpoint(save_path)
            self.__last_save += self.__interval
