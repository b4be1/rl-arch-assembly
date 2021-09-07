from typing import TypeVar, Generic, Iterable, Dict, Union, Tuple, Any, Optional

import gym
import numpy as np

from assembly_gym.environment.generic import Environment
from task.rewards.goal.goal_reward import GoalReward
from task.sensors.invertible_sensor import InvertibleSensor

EnvType = TypeVar("EnvType")
GoalType = TypeVar("GoalType")


class GoalEnvWrapper(gym.GoalEnv, Generic[EnvType, GoalType]):
    def __init__(self, wrapped_env: EnvType, achieved_goal_sensor: InvertibleSensor[EnvType, GoalType],
                 desired_goal_sensor: InvertibleSensor[EnvType, GoalType], goal_rewards: Iterable[GoalReward]):
        self.__wrapped_env: EnvType = wrapped_env
        self.__achieved_goal_sensor: InvertibleSensor[EnvType] = achieved_goal_sensor
        self.__desired_goal_sensor: InvertibleSensor[EnvType] = desired_goal_sensor
        self.__achieved_goal_obs_space: Optional[gym.spaces.Space] = None
        self.__desired_goal_obs_space: Optional[gym.spaces.Space] = None
        self.__goal_rewards: Iterable[GoalReward] = goal_rewards

    def initialize(self, env: Environment, auto_restart: bool) -> None:
        self.__wrapped_env.initialize(env, auto_restart)
        self.__achieved_goal_obs_space = gym.spaces.Dict(self.__achieved_goal_sensor.initialize(self.__wrapped_env))
        self.__desired_goal_obs_space = gym.spaces.Dict(self.__desired_goal_sensor.initialize(self.__wrapped_env))
        for goal_reward in self.__goal_rewards:
            goal_reward.initialize(self.__wrapped_env)

    def reset(self) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        wrapped_env_obs = self.__wrapped_env.reset()
        achieved_goal_obs = self.__achieved_goal_sensor.reset()
        desired_goal_obs = self.__desired_goal_sensor.reset()
        for goal_reward in self.__goal_rewards:
            goal_reward.reset()
        return {"observation": wrapped_env_obs, "achieved_goal": achieved_goal_obs, "desired_goal": desired_goal_obs}

    def step(self, action: Dict[str, np.ndarray]) \
            -> Tuple[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]], float, bool, Dict[str, Any]]:
        wrapped_env_obs, wrapped_env_reward, done, info = self.__wrapped_env.step(action)

        # Store information that is necessary to calculate the reward afterwards in self.compute_reward. The information
        # is stored in info since by the definition of gym.GoalEnv, self.compute_reward only has access to the
        # achieved_goal, desired_goal, and the info dict.
        goal_rewards_information_list = [goal_reward.information_to_store() for goal_reward in self.__goal_rewards]
        goal_rewards_info = {key: val for dic in goal_rewards_information_list for key, val in dic.items()}
        goal_rewards_info["done"] = done
        goal_env_wrapper_info = {"wrapped_env_reward": wrapped_env_reward, "goal_rewards_info": goal_rewards_info}
        info["goal_env_wrapper_info"] = goal_env_wrapper_info

        achieved_goal_obs = self.__achieved_goal_sensor.observe()
        desired_goal_obs = self.__desired_goal_sensor.observe()
        obs = {"observation": wrapped_env_obs, "achieved_goal": achieved_goal_obs, "desired_goal": desired_goal_obs}

        reward = self.compute_reward(achieved_goal_obs, desired_goal_obs, info)
        info["goal_reward"] = reward - wrapped_env_reward

        return obs, reward, done, info

    def compute_reward(self, achieved_goal_obs: Dict[str, np.ndarray], desired_goal_obs: Dict[str, np.ndarray], info) \
            -> float:
        goal_env_info = info["goal_env_wrapper_info"]
        achieved_goal = self.__achieved_goal_sensor.observation_to_state(achieved_goal_obs)
        desired_goal = self.__desired_goal_sensor.observation_to_state(desired_goal_obs)
        wrapped_env_reward = goal_env_info["wrapped_env_reward"]
        goal_reward = sum(goal_reward.calculate_reward(achieved_goal, desired_goal, goal_env_info["goal_rewards_info"])
                          for goal_reward in self.__goal_rewards)
        return wrapped_env_reward + goal_reward

    @property
    def unwrapped(self) -> EnvType:
        return self.__wrapped_env

    def render(self, mode: str = "human") -> None:
        self.__wrapped_env.render(mode)

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.__wrapped_env.action_space

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({"observation": self.__wrapped_env.observation_space,
                                "achieved_goal": self.__achieved_goal_obs_space,
                                "desired_goal": self.__desired_goal_obs_space})
