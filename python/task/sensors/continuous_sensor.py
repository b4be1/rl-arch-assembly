from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, Tuple, List, Optional, TYPE_CHECKING

import gym
import numpy as np

from .sensor import Sensor
from task.normalizer import Normalizer

if TYPE_CHECKING:
    from task import BaseTask

TaskType = TypeVar("TaskType", bound="BaseTask")


class ContinuousSensor(Sensor[TaskType], Generic[TaskType], ABC):
    """
    An abstract base class for sensors with continuous outputs. Every observation is normalized (and clipped) to
    [-1, 1].
    """

    def __init__(self, normalize: bool = True, clip: bool = False, **kwargs):
        super(ContinuousSensor, self).__init__(**kwargs)
        self.__normalize = normalize
        self.__normalizers: Dict[str, Normalizer] = {}
        self.__clip = clip
        self._limits: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None

    def _get_observation_spaces(self) -> Dict[str, gym.spaces.Box]:
        """
        Create the observation observation spaces as gym Box spaces.

        :return:        a dictionary that maps names of observations to corresponding gym Box spaces
        """

        assert self._limits is not None, "Sensor must be initialized first"
        if self.__normalize:
            shapes = {name: bound[0].shape for name, bound in self._limits.items()}
            return {name: gym.spaces.Box(-np.ones(shape, dtype=np.float32), np.ones(shape, dtype=np.float32))
                    for name, shape in shapes.items()}
        else:
            return {name: gym.spaces.Box(lower_limit.astype(np.float32), upper_limit.astype(np.float32))
                    for name, (lower_limit, upper_limit) in self._limits.items()}

    def _initialize(self):
        self._limits = self._get_limits()
        for lower, upper in self._limits.values():
            assert lower.shape == upper.shape, \
                "Shape of lower limits {} does not match shape of upper limits {}".format(lower.shape, upper.shape)
            assert np.all(lower < upper), \
                "Lower limits must be smaller than upper limits, but found lower limits {} and upper limits {}".format(
                    lower, upper
                )
        for name, (lower, upper) in self._limits.items():
            self.__normalizers[name] = Normalizer(lower, upper)

    def reset(self) -> Dict[str, np.ndarray]:
        observations_unnormalized = self._reset_unnormalized()
        return self._maybe_normalize_clip_obs_dict(observations_unnormalized)

    def observe(self) -> Dict[str, np.ndarray]:
        observations_unnormalized = self._observe_unnormalized()
        return self._maybe_normalize_clip_obs_dict(observations_unnormalized)

    def _maybe_normalize_clip_obs_dict(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalizes and clips an observation dictionary to [-1, 1].

        :param observations:        a dictionary that maps observation names to (unnormalized) values
        :return:                    the input dictionary where every observation is normalized and clipped to [-1, 1]
                                    according to the limits in self.__limits
        """

        assert self._limits is not None, "Sensor must be initialized first"
        if self.__clip:
            observations = {name: self._clip(observation, self._limits[name][0], self._limits[name][1])
                            for name, observation in observations.items()}
        if self.__normalize:
            observations = {name: self.__normalizers[name].normalize(observation)
                            for name, observation in observations.items()}
        return observations

    @staticmethod
    def _clip(obs: np.ndarray, lower_limit: np.ndarray, upper_limit: np.ndarray) -> np.ndarray:
        """
        Clips the (normalized) observations to lie in [lower_limit, upper_limit].

        :param obs:             an observation as an array
        :param lower_limit:     the lower limit
        :param upper_limit:     the upper limit
        :return:                the observation clipped to [lower_limit, upper_limit]
        """
        return np.maximum(lower_limit, np.minimum(obs, upper_limit))

    @abstractmethod
    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns the minimum and maximum (unnormalized) values that the sensor should observe. Every value outputted by
        the sensor that is smaller/larger than this minimum/maximum will be clipped to -1/1. Must be implemented by
        subclasses.

        :return:        a dictionary that maps each observation name to a tuple (lower, upper), where lower/upper is
                        an array of the minimum/maximum values that the sensor should observe for this observation
        """
        raise NotImplementedError("_get_limits must be overwritten")

    @abstractmethod
    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        """
        Resets the sensor (called at the beginning of each episode). Must be implemented by subclasses.

        :return:        the first observation of the episode as a dictionary that maps observation names to
                        (unnormalized) values
        """
        raise NotImplementedError("_reset_unnormalized must be overwritten")

    @abstractmethod
    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        """
        Calculates the (unnormalized) observations. Must be implemented by subclasses.

        :return:        a dictionary that maps observation names to (unnormalized) values
        """
        raise NotImplementedError("_observe_unnormalized must be overwritten")

    @property
    def names(self) -> List[str]:
        """
        Returns the names of all observations that this sensor provides.

        :return:        a list of names of all observations that this sensor provides
        """
        return list(self._limits.keys())
