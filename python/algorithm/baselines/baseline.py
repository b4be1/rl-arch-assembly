import logging

import numpy as np
from pathlib import Path
from typing import Dict, Type, Any, Union

import gym
import stable_baselines3
import stable_baselines3.common.utils
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

from algorithm import Algorithm


class _LoggerConfigureCallback(BaseCallback):
    def __init__(self, summary_writer: SummaryWriter, log_path: Path):
        super(_LoggerConfigureCallback, self).__init__()
        self.__summary_writer = summary_writer
        self.__log_path = log_path

    def _on_training_start(self):
        current_logger: logger.Logger = logger.Logger.CURRENT
        tensorboard = logger.TensorBoardOutputFormat(str(self.__log_path / "tensorboard"))
        tensorboard.writer = self.__summary_writer
        current_logger.output_formats.append(tensorboard)

    def _on_step(self) -> bool:
        return True


class Baseline(Algorithm):
    def __init__(self, algorithm_class: Type[BaseAlgorithm], model_parameters: Dict[str, Any]):
        super(Baseline, self).__init__()
        self._algorithm_class = algorithm_class
        self._model = None
        self._logger = logging.getLogger("training")
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(logging.StreamHandler())
        self._model_parameters = model_parameters

    def _load_checkpoint(self, path: Path, env: Union[gym.Env, VecEnv]):
        self._model = self._algorithm_class.load(str(path), env)

    def _initialize_new_model(self, env: Union[gym.Env, VecEnv]):
        self._model = self._algorithm_class(env=env, **self._model_parameters)

    @staticmethod
    def class_from_name(name: str) -> Type[stable_baselines3.common.base_class.BaseAlgorithm]:
        if name.upper() == "TD3":
            return stable_baselines3.TD3
        elif name.upper() == "A2C":
            return stable_baselines3.A2C
        elif name.upper() == "SAC":
            return stable_baselines3.SAC
        else:
            raise RuntimeError("Unknown model: {}".format(name))

    def get_model_parameters(self) -> Dict[str, Any]:
        return self._model_parameters

    def _train(self, log_path: Path, total_timesteps: int, train_parameters: Dict, summary_writer: SummaryWriter):
        self._model.learn(log_interval=1, callback=_LoggerConfigureCallback(summary_writer, log_path),
                          total_timesteps=total_timesteps, **train_parameters)

    def predict(self, obs: np.ndarray, evaluation_mode: bool = False) -> np.ndarray:
        return self._model.predict(obs, deterministic=evaluation_mode)[0]

    def save_checkpoint(self, path: Path):
        self._model.save(path)

    @Algorithm.num_workers.setter
    def num_workers(self, value: int):
        if self._algorithm_class == stable_baselines3.A2C:
            Algorithm.num_workers.fset(self, value)
        else:
            Algorithm.num_workers.fset(self, 1)
