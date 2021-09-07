from abc import abstractmethod, ABC

import gym
import numpy as np

from typing import TypeVar, Generic

GoalType = TypeVar("GoalType")


class GoalSensor(Generic[GoalType], ABC):
    @abstractmethod
    def observe(self, goal_state: GoalType) -> np.ndarray:
        pass

    @abstractmethod
    def get_state(self, goal_observation: np.ndarray) -> GoalType:
        pass

    @abstractmethod
    def _get_goal_observation_space(self) -> gym.spaces.Box:
        pass

    @property
    def goal_observation_space(self) -> gym.spaces.Box:
        return self._get_goal_observation_space()
