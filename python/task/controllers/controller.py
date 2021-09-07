from abc import abstractmethod, ABC
from typing import Optional, Tuple

import gym
import numpy as np

from task import BaseTask
from task.normalizer import Normalizer


class Controller(ABC):
    """
    An abstract base class for the different types of controllers (e.g. torque or velocity controller).
    """

    def __init__(self, name: str):
        self.__name: str = name
        self.__task: Optional[BaseTask] = None
        self.__action_limits_lower: Optional[np.ndarray] = None
        self.__action_limits_upper: Optional[np.ndarray] = None
        self.__normalizer: Optional[Normalizer] = None

    def initialize(self, task: BaseTask) -> gym.spaces.Space:
        """
        Initializes the controller with for a given task.
        """
        self.__task: BaseTask = task
        self.__action_limits_lower, self.__action_limits_upper = self._initialize(task)
        assert len(self.__action_limits_lower) == len(self.__action_limits_upper), \
            "Sizes of action limits do not match ({} vs {})".format(
                len(self.__action_limits_upper), len(self.__action_limits_upper))
        self.__normalizer = Normalizer(self.__action_limits_lower, self.__action_limits_upper)
        lower = -np.ones_like(self.__action_limits_lower, dtype=np.float32)
        upper = np.ones_like(self.__action_limits_upper, dtype=np.float32)
        return gym.spaces.Box(lower, upper, shape=(self.__action_limits_lower.size, ))

    def actuate(self, action_normalized: np.ndarray) -> None:
        """
        Actuate the robot component (e.g. arm or gripper) according to a given action.

        :param action_normalized:       a normalized action as (N, ) array (in [-1, 1])
        """
        action_clipped = np.maximum(np.minimum(action_normalized, np.ones_like(action_normalized)),
                                    -np.ones_like(action_normalized))
        self._actuate_denormalized(self.__normalizer.denormalize(action_clipped))

    @abstractmethod
    def _actuate_denormalized(self, action: np.ndarray) -> None:
        """
        Actuates the robot according to a given (unnormalized) action. Must be implemented by the concrete instantiation
        of the controller.

        :param action:          an action as (N, ) array
        """
        pass

    @abstractmethod
    def _initialize(self, task: BaseTask) -> Tuple[np.ndarray, np.ndarray]:
        """
        Configure the joints of the robot to be used with the controller. Must be implemented by the concrete
        instantiation of the controller.

        :param task:    the task in which the controller is used
        :return:        a (lower, upper)-tuple of the lower and upper limits of the unnormalized action
        """
        pass

    @property
    def task(self) -> BaseTask:
        return self.__task

    @property
    def name(self) -> str:
        return self.__name

    @property
    def action_limits_lower(self) -> np.ndarray:
        return self.__action_limits_lower

    @property
    def action_limits_upper(self) -> np.ndarray:
        return self.__action_limits_upper

    def __repr__(self) -> str:
        return "Controller {}".format(self.__name)
