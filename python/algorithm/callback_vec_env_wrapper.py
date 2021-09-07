from typing import List, Sequence, Optional, Union, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from algorithm.callbacks import Callback


class CallbackVecEnvWrapper(VecEnv):
    def __init__(self, wrapped_env: VecEnv):
        super(CallbackVecEnvWrapper, self).__init__(
            wrapped_env.num_envs, wrapped_env.observation_space, wrapped_env.action_space)
        self.__wrapped_env = wrapped_env
        self.callbacks: List[Callback] = []
        self.__current_time_step = None

    def step(self, action: Tuple[np.ndarray]):
        obs, reward, done, info = self.__wrapped_env.step(action)
        for cb in self.callbacks:
            cb.on_step(self.__current_time_step, action, obs, reward, done, info)
        self.__current_time_step += len(obs)
        return obs, reward, done, info

    def reset(self):
        if self.__current_time_step is not None:
            for cb in self.callbacks:
                cb.on_episode_end()
        else:
            self.__current_time_step = 0
        obs = self.__wrapped_env.reset()
        for cb in self.callbacks:
            cb.on_episode_start(obs)
        return obs

    def render(self, mode='human'):
        return self.__wrapped_env.render(mode)

    def close(self):
        self.__wrapped_env.close()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.__wrapped_env.env_method(method_name, method_args, indices, **method_kwargs)

    def get_attr(self, attr_name, indices=None):
        return self.__wrapped_env.get_attr(attr_name, indices)

    def get_images(self) -> Sequence[np.ndarray]:
        return self.__wrapped_env.get_images()

    def getattr_depth_check(self, name, already_found):
        return self.__wrapped_env.getattr_depth_check(name, already_found)

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self.__wrapped_env.seed(seed)

    def set_attr(self, attr_name, value, indices=None):
        return self.__wrapped_env.set_attr(attr_name, value, indices)

    def step_async(self, actions):
        return self.__wrapped_env.step_async(actions)

    def step_wait(self):
        return self.__wrapped_env.step_wait()

    @property
    def wrapped_env(self) -> VecEnv:
        return self.__wrapped_env

    @property
    def unwrapped(self):
        return self.__wrapped_env.unwrapped
