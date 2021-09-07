from typing import Dict, Tuple, TypeVar, Generic, Callable

import numpy as np

from .continuous_sensor import ContinuousSensor
from task import BaseTask

EnvType = TypeVar("EnvType", bound=BaseTask)


class LambdaContinuousSensor(ContinuousSensor[EnvType], Generic[EnvType]):
    """
    A sensor for continuous observations (results in a Dict space of Box spaces) that uses a custom function to
    calculate the observations.
    """

    def __init__(self, lmbda: Callable[[EnvType], Dict[str, np.ndarray]],
                 limits: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 normalize: bool = True, clip: bool = True, **kwargs):
        """
        :param lmbda:           a function that takes the gym environment as input and produces an observation
                                dictionary for the current time step (i.e. a mapping from observation names to values)
        :param limits:          a dictionary that maps observation names to lower and upper bounds for the observation
                                (i.e. limits[name] = (limit_lower, limit_upper)); Note that limits must contain lower
                                and upper bounds for all observations returned by lmbda
        :param normalize:       whether the observations returned by lmbda should be normalized to [-1, 1]
        :param clip:            whether values below/above the lower/upper limit should be clipped
        """
        super().__init__(normalize=normalize, clip=clip, **kwargs)
        self.__lmbda = lmbda
        self.__limits = limits

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return self.__limits

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self._observe_unnormalized()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.__lmbda(self.task)
