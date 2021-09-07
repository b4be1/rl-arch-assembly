import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Iterable, Any, Callable, Union

import numpy as np

import gym
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

from algorithm.callback_env_wrapper import CallbackEnvWrapper
from .callback_goal_env_wrapper import CallbackGoalEnvWrapper
from .callback_vec_env_wrapper import CallbackVecEnvWrapper
from .callbacks import Callback, ProgressCallback, SaveModelCallback, TensorboardCallback
from task.wrappers import FlattenWrapper, FlattenWrapperGoalEnv


class Algorithm(ABC):
    def __init__(self):
        self.__num_workers = 1
        self.__wrapped_env: Optional[Union[CallbackEnvWrapper, CallbackVecEnvWrapper]] = None

    def train(self, log_path: Path, total_timesteps: int, train_parameters: Dict,
              callbacks: Optional[Iterable[Callback]] = None):
        if callbacks is None:
            callbacks = []
        else:
            callbacks = list(callbacks)
        summary_writer = SummaryWriter(log_dir=str(log_path / "tensorboard"))
        callbacks.append(ProgressCallback(1000, logging.getLogger("training")))
        callbacks.append(SaveModelCallback(100000, log_path))
        callbacks.append(TensorboardCallback(summary_writer))
        print("========================= Starting training =========================")
        for cb in callbacks:
            cb.init_training(self, total_timesteps)
        for cb in callbacks:
            cb.on_training_start()
        self.__wrapped_env.callbacks = callbacks
        self._train(log_path, total_timesteps, train_parameters, summary_writer)
        for cb in callbacks:
            cb.on_training_end()
        print("========================= Done training =========================")

    @abstractmethod
    def _train(self, log_path: Path, total_timesteps: int, train_parameters: Dict, summary_writer: SummaryWriter):
        pass

    @abstractmethod
    def predict(self, obs: np.ndarray, evaluation_mode: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def save_checkpoint(self, path: Path):
        pass

    def load_checkpoint(self, path: Path, env_factory: Callable[[], gym.Env]):
        self.__create_env(env_factory)
        self._load_checkpoint(path, self.__wrapped_env)

    @abstractmethod
    def _load_checkpoint(self, path: Path, env: Union[gym.Env, VecEnv]):
        pass

    @abstractmethod
    def _initialize_new_model(self, env: Union[gym.Env, VecEnv]):
        pass

    def shutdown(self):
        self.__wrapped_env.close()

    def __create_env(self, env_factory: Callable[[], gym.Env]):
        if self.__num_workers == 1:
            env_unflattened = env_factory()
            if isinstance(env_unflattened, gym.GoalEnv):
                env = FlattenWrapperGoalEnv(env_unflattened)
                self.__wrapped_env = CallbackGoalEnvWrapper(env)
            else:
                env = FlattenWrapper(env_unflattened)
                self.__wrapped_env = CallbackEnvWrapper(env)
        else:
            def factory(_inner_fact=env_factory):
                _env_unflattened = _inner_fact()
                if isinstance(_env_unflattened, gym.GoalEnv):
                    _env = FlattenWrapperGoalEnv(_env_unflattened)
                else:
                    _env = FlattenWrapper(_env_unflattened)
                return _env

            env = SubprocVecEnv([factory for _ in range(self.num_workers)])
            self.__wrapped_env = CallbackVecEnvWrapper(env)

    def initialize_new_model(self, env_factory: Callable[[], gym.Env]):
        self.__create_env(env_factory)
        self._initialize_new_model(self.__wrapped_env)

    def get_model_parameters(self) -> Dict[str, Any]:
        return {}

    def visualize_episode(self, env: gym.Env, render: bool = False, time_per_step: Optional[float] = None,
                          step_callback: Optional[Callable[[np.ndarray, float, bool, Dict], None]] = None,
                          reset_callback: Optional[Callable[[np.ndarray], None]] = None) -> float:
        done = False
        total_reward = 0
        obs = env.reset()
        if reset_callback is not None:
            reset_callback(obs)
        next_step = time.time()
        while not done:
            if time_per_step is not None:
                time.sleep(max(0.0, next_step - time.time()))
                next_step += time_per_step
            action = self.predict(obs, evaluation_mode=True)
            obs, reward, done, info = env.step(action)
            if step_callback is not None:
                step_callback(obs, reward, done, info)
            total_reward += reward
            if render:
                env.render()
        return total_reward

    @property
    def model_parameters(self):
        return self.get_model_parameters()

    @property
    def num_workers(self) -> int:
        return self.__num_workers

    @num_workers.setter
    def num_workers(self, value: int):
        self.__num_workers = value
