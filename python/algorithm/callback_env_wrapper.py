from typing import List

import gym

import numpy as np

from algorithm.callbacks import Callback


class CallbackEnvWrapper(gym.Env):
    def __init__(self, wrapped_env: gym.Env):
        self._wrapped_env = wrapped_env
        self.callbacks: List[Callback] = []
        self.__current_time_step = None

    def step(self, action: np.ndarray):
        obs, reward, done, info = self._wrapped_env.step(action)
        for cb in self.callbacks:
            cb.on_step(self.__current_time_step, (action, ), (obs, ), (reward, ), (done, ), (info, ))
        self.__current_time_step += 1
        return obs, reward, done, info

    def reset(self):
        if self.__current_time_step is not None:
            for cb in self.callbacks:
                cb.on_episode_end()
        else:
            self.__current_time_step = 0
        obs = self._wrapped_env.reset()
        for cb in self.callbacks:
            cb.on_episode_start((obs, ))
        return obs

    def render(self, mode='human'):
        return self._wrapped_env.render(mode)

    def close(self):
        self._wrapped_env.close()

    @property
    def wrapped_env(self) -> gym.Env:
        return self._wrapped_env

    @property
    def unwrapped(self):
        return self._wrapped_env.unwrapped

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space
