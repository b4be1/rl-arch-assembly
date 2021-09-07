from typing import Union, Dict, Any, Tuple, Sequence

import gym
import numpy as np

from .dict_flattener import DictFlattener


class FlattenWrapperGoalEnv(gym.GoalEnv):
    """
    A wrapper for gym.GoalEnv environments that converts the gym.spaces.Dict action space and all of the gym.spaces.Dict
    observation spaces "observation", "achieved_goal", and "desired_goal" of the GoalEnv observation to gym.spaces.Box.
    Needed since stable-baselines3 does not support Dict spaces.
    """

    def __init__(self, wrapped_env: gym.GoalEnv):
        assert isinstance(wrapped_env.action_space, gym.spaces.Dict), "The action space must be of type Dict"
        assert isinstance(wrapped_env.observation_space, gym.spaces.Dict), "The observation space must be of type Dict"
        for key in ["observation", "achieved_goal", "desired_goal"]:
            assert key in wrapped_env.observation_space.spaces, \
                "The observation of a GoalEnv must contain the key \"{}\"".format(key)

        self.__wrapped_env = wrapped_env
        self.__action_flattener = DictFlattener(self.__wrapped_env.action_space)
        self.__observation_flattener = DictFlattener(self.__wrapped_env.observation_space.spaces["observation"])
        self.__achieved_goal_flattener = DictFlattener(self.__wrapped_env.observation_space.spaces["achieved_goal"])
        self.__desired_goal_flattener = DictFlattener(self.__wrapped_env.observation_space.spaces["desired_goal"])

    def reset(self) -> Dict[str, np.ndarray]:
        obs_dict = self.__wrapped_env.reset()
        obs = self.pack_observation(obs_dict)
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        action_dict = self.unpack_action(action)

        obs_dict, reward, done, info = self.__wrapped_env.step(action_dict)
        obs = self.pack_observation(obs_dict)
        return obs, reward, done, info

    def pack_observation(self, obs: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
        observation_packed = self.__observation_flattener.pack_dict(obs["observation"])
        achieved_goal_packed = self.__achieved_goal_flattener.pack_dict(obs["achieved_goal"])
        desired_goal_packed = self.__desired_goal_flattener.pack_dict(obs["desired_goal"])
        return {"observation": observation_packed, "achieved_goal": achieved_goal_packed,
                "desired_goal": desired_goal_packed}

    def unpack_observation(self, packed_obs: np.ndarray) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        observation_unpacked = self.__observation_flattener.unpack_dict(packed_obs["observation"])
        achieved_goal_unpacked = self.__achieved_goal_flattener.unpack_dict(packed_obs["achieved_goal"])
        desired_goal_unpacked = self.__desired_goal_flattener.unpack_dict(packed_obs["desired_goal"])
        return {"observation": observation_unpacked, "achieved_goal": achieved_goal_unpacked,
                "desired_goal": desired_goal_unpacked}

    def pack_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        return self.__action_flattener.pack_dict(action)

    def unpack_action(self, packed_action: np.ndarray) -> Dict[str, np.ndarray]:
        return self.__action_flattener.unpack_dict(packed_action)

    def render(self, mode: str = "human") -> None:
        self.__wrapped_env.render(mode)

    def close(self) -> None:
        self.__wrapped_env.close()

    def compute_reward(self, achieved_goals: np.ndarray, desired_goals: np.ndarray, infos: Sequence[Dict[str, Any]]) \
            -> np.ndarray:
        # TODO: Multiple observations in achieved and desired goal really shouldn't be handled by the FlattenWrapper
        rewards = []
        for achieved, desired, info in zip(achieved_goals, desired_goals, infos):
            achieved_unpacked = self.__achieved_goal_flattener.unpack_dict(achieved)
            desired_unpacked = self.__desired_goal_flattener.unpack_dict(desired)
            rewards.append(self.__wrapped_env.compute_reward(achieved_unpacked, desired_unpacked, info))
        return np.array(rewards)

    @property
    def wrapped_env(self) -> gym.Env:
        return self.__wrapped_env

    @property
    def unwrapped(self):
        return self.__wrapped_env.unwrapped

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.__action_flattener.flattened_space

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({"observation": self.__observation_flattener.flattened_space,
                                "achieved_goal": self.__achieved_goal_flattener.flattened_space,
                                "desired_goal": self.__desired_goal_flattener.flattened_space})
