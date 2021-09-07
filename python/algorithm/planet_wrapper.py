import numpy as np
from typing import Dict, Any, Optional, Union

from pathlib import Path

import gym
import torch
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

from algorithm.planet import Planet
from .algorithm import Algorithm


class PlanetWrapper(Algorithm):
    def __init__(self, model_parameters: Dict[str, Any]):
        super(PlanetWrapper, self).__init__()
        self.__planet: Optional[Planet] = None
        self.__model_parameters = model_parameters
        self.__prev_belief = None
        self.__prev_action = None
        self.__prev_state = None

    def _train(self, log_path: Path, total_timesteps: int, train_parameters: Dict, summary_writer: SummaryWriter):
        self.__planet.train(total_timesteps, summary_writer=summary_writer, **train_parameters)

    def predict(self, obs: np.ndarray, evaluation_mode: bool = False) -> np.ndarray:
        if self.__prev_belief is None:
            self.__prev_belief = torch.zeros(1, self.__planet.belief_size, device=self.__planet.device)
            self.__prev_action = torch.zeros(1, self.__planet.state_size, device=self.__planet.device)
            self.__prev_action = torch.zeros(1, self.__planet.action_size, device=self.__planet.device)
        self.__prev_belief, self.__prev_state, self.__prev_action, observation, reward, done = \
            self.__planet.update_belief_and_act(
                self.__prev_belief, self.__prev_action, self.__prev_action,
                torch.from_numpy(obs).to(device=self.__planet.device), self.__planet.min_action,
                self.__planet.max_action)
        if done:
            self.__prev_belief = self.__prev_state = self.__prev_action = None
        return self.__prev_action

    def save_checkpoint(self, path: Path):
        self.__planet.save_checkpoint(path, store_replay_buffer=False)

    def _load_checkpoint(self, path: Path, env: Union[gym.Env, VecEnv]):
        self.__planet = Planet(env, symbolic_env=True, **self.__model_parameters)
        self.__planet.load_checkpoint(path)

    def _initialize_new_model(self, env: Union[gym.Env, VecEnv]):
        self.__planet = Planet(env, symbolic_env=True, **self.__model_parameters)
