from typing import Dict, TypeVar, Generic, Callable

import numpy as np

from task import BaseTask
from .discrete_sensor import DiscreteSensor

EnvType = TypeVar("EnvType", bound=BaseTask)


class LambdaDiscreteSensor(DiscreteSensor[EnvType], Generic[EnvType]):
    """
    A sensor for discrete observations (results in a Dict space of MultiDiscrete spaces) that uses a custom function to
    calculate the observations.
    """

    def __init__(self, lmbda: Callable[[EnvType], Dict[str, np.ndarray]],
                 nr_values: Dict[str, np.ndarray], **kwargs):
        """
        :param lmbda:           a function that takes the gym environment as input and produces an observation
                                dictionary for the current time step (i.e. a mapping from observation names to values)
        :param nr_values:       a dictionary that maps observation names to to the number of possible values for these
                                observations; Note that nr_values must contain one entry for all observations returned
                                by lmbda
        """
        super().__init__(**kwargs)
        self.__lmbda = lmbda
        self.__nr_values = nr_values

    def _get_nr_values(self) -> Dict[str, np.ndarray]:
        return self.__nr_values

    def reset(self) -> Dict[str, np.ndarray]:
        return self.observe()

    def observe(self) -> Dict[str, np.ndarray]:
        return self.__lmbda(self.task)
