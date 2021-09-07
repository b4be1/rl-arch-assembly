from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Sequence, Optional

import gym.spaces
import numpy as np

from .sensor import Sensor

TaskType = TypeVar("TaskType")


class DiscreteSensor(Sensor[TaskType], Generic[TaskType], ABC):
    """
    An abstract base class for sensors with discrete outputs.
    """

    def __int__(self, **kwargs):
        super(DiscreteSensor, self).__init__(**kwargs)
        self.__nr_values: Optional[Dict[str, np.ndarray]] = None

    def _get_observation_spaces(self) -> Dict[str, gym.spaces.MultiDiscrete]:
        """
        Create the observation observation spaces as gym MultiDiscrete spaces.

        :return:        a dictionary that maps names of observations to corresponding gym Box spaces
        """
        shapes = {name: bound[0].shape for name, bound in self.__nr_values.items()}
        return {name: gym.spaces.MultiDiscrete(self.__nr_values) for name, shape in shapes.items()}

    def _initialize(self):
        self.__nr_values = self._get_nr_values()
        for n in self.__nr_values.values():
            assert np.all(0 <= n), "The number of possible values cannot be negative, found: {}".format(n)

    @abstractmethod
    def _get_nr_values(self) -> Dict[str, np.ndarray]:
        """
        Returns an array that contains the number of possible values for each entry of all observations.

        :return:        a dictionary that maps observation names to an array that contains the number of possible values
                        for each entry of the observation
        """
        raise NotImplementedError("_get_limits must be overwritten")

    @property
    def names(self) -> Sequence[str]:
        """
        Returns the names of all observations that this sensor provides.

        :return:        a list of names of all observations that this sensor provides
        """
        return list(self.__nr_values.keys())
