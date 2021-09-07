from abc import abstractmethod
from typing import Dict, Optional, TypeVar, Generic, TYPE_CHECKING

import numpy as np
import gym.spaces

if TYPE_CHECKING:
    from task import BaseTask

TaskType = TypeVar("TaskType", bound="BaseTask")


class Sensor(Generic[TaskType]):
    """
    An abstract base class for sensors. Sensors produce the observations for the agent.
    """

    def __init__(self, **kwargs):
        super(Sensor, self).__init__(**kwargs)
        self.__task: Optional[TaskType] = None

    def initialize(self, task: TaskType) -> Dict[str, gym.spaces.Space]:
        """
        Initializes the sensor.

        :param task:         the gym environment that the sensor should be used for
        :return:            a dictionary that maps names to the observation spaces that the sensor provides; note that
                            a sensor might provide multiple independent observations (e.g. linear and angular
                            velocities), which results in the dictionary containing multiple observation spaces
        """
        self.__task: TaskType = task
        self._initialize()
        return self._get_observation_spaces()

    @abstractmethod
    def _initialize(self) -> None:
        """
        Initializes the concrete instance of the sensor. Must be overwritten by subclasses.

        :return:            a dictionary that maps names to the observation spaces that the sensor provides
        """
        raise NotImplementedError("Method _initialize must be overwritten")

    @abstractmethod
    def _get_observation_spaces(self) -> Dict[str, gym.spaces.Space]:
        raise NotImplementedError("Method _get_observation_space must be overwritten")

    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the sensor at the end of the episode.

        :return:            the observation at the beginning of the episode
        """
        raise NotImplementedError("Method reset must be overwritten")

    @abstractmethod
    def observe(self) -> Dict[str, np.ndarray]:
        """
        Calculates the agent's observations for the current time step.

        :return:            a dictionary that maps observation names to the corresponding values
        """
        raise NotImplementedError("Method observe must be overwritten")

    @property
    def task(self) -> Optional[TaskType]:
        """
        Returns the task that the sensor operates on.

        :return:            the task that the sensor operates on
        """
        return self.__task

    def __repr__(self):
        return type(self).__name__ + "()"
