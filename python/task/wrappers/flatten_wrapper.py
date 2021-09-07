from typing import Dict, Tuple

import gym
import numpy as np

from .dict_flattener import DictFlattener


class FlattenWrapper(gym.Env):
    """
    A wrapper for gym environments that converts gym.spaces.Dict observation and action spaces to gym.spaces.Box.
    Needed since stable-baselines3 does not support Dict spaces.
    """

    def __init__(self, wrapped_env: gym.Env):
        assert isinstance(wrapped_env.action_space, gym.spaces.Dict), "The action space must be of type Dict"
        assert isinstance(wrapped_env.observation_space, gym.spaces.Dict), "The observation space must be of type Dict"
        self.__wrapped_env = wrapped_env
        self.__action_flattener = DictFlattener(self.__wrapped_env.action_space)
        self.__observation_flattener = DictFlattener(self.__wrapped_env.observation_space)

    def reset(self) -> np.ndarray:
        obs_dict = self.__wrapped_env.reset()
        obs = self._pack_observation(obs_dict)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action_dict = self._unpack_action(action)

        obs_dict, reward, done, info = self.__wrapped_env.step(action_dict)
        obs = self._pack_observation(obs_dict)
        return obs, reward, done, info

    def _pack_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self.__observation_flattener.pack_dict(obs)

    def _unpack_observation(self, packed_obs: np.ndarray) -> Dict[str, np.ndarray]:
        return self.__observation_flattener.unpack_dict(packed_obs)

    def _pack_action(self, act: Dict[str, np.ndarray]) -> np.ndarray:
        return self.__action_flattener.pack_dict(act)

    def _unpack_action(self, packed_act: np.ndarray) -> Dict[str, np.ndarray]:
        return self.__action_flattener.unpack_dict(packed_act)

    def render(self, mode='human') -> None:
        self.__wrapped_env.render(mode)

    def close(self) -> None:
        self.__wrapped_env.close()

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.__action_flattener.flattened_space

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self.__observation_flattener.flattened_space

    @property
    def wrapped_env(self) -> gym.Env:
        return self.__wrapped_env

    @property
    def unwrapped(self):
        return self.__wrapped_env.unwrapped
