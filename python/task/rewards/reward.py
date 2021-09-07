from abc import abstractmethod
from typing import TypeVar, Generic, Optional, Callable

from task import BaseTask

TaskType = TypeVar("TaskType", bound=BaseTask)


class Reward(Generic[TaskType]):
    """
    An abstract base class for rewards. Implementing classes should make sure that the reward lies in [-1, 0], so that
    all rewards have the same scaling and weight (scaling) factors can be chosen easier. These scaling factors are used
    to weigh different rewards relative to each other.
    """

    def __init__(self, name: str, intermediate_timestep_reward_scale: float,
                 final_timestep_reward_scale: Optional[float] = None, normalize: bool = True, clip: bool = False,
                 abbreviated_name: Optional[str] = None, reward_condition: Optional[Callable[[], bool]] = None):
        """
        :param name:                                the name of the reward (to be used as key in info dictionary
                                                    returned by the gym environment every step)
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        :param normalize:                           whether the (unscaled) reward should be normalized to [-1, 0]
        :param clip:                                whether the (unnormalized, unscaled) reward should be clipped to not
                                                    go lower than the value returned by _get_min_reward_unnormalized()
        :param abbreviated_name:                    an abbreviation of the name of the reward that is displayed with the
                                                    value of the reward at the end of each episode (by the gym
                                                    environment)
        :param reward_condition:                    a custom condition that specifies if the reward is active (i.e. the
                                                    reward is only calculated if reward_condition() is True, otherwise
                                                    the calculate_reward() returns 0)
        """
        self.__name: str = name
        if abbreviated_name is not None:
            self.__name_abbreviation = abbreviated_name
        else:
            self.__name_abbreviation = name
        self.__intermediate_timestep_reward_scale: float = intermediate_timestep_reward_scale
        if final_timestep_reward_scale is None:
            self.__final_timestep_reward_scale: float = intermediate_timestep_reward_scale
        else:
            self.__final_timestep_reward_scale: float = final_timestep_reward_scale
        self.__normalize: bool = normalize
        self.__clip: bool = clip
        self.__task: Optional[TaskType] = None
        self.__min_reward: Optional[float] = None
        self.__reward_condition: Optional[Callable[[], bool]] = reward_condition

    def initialize(self, task: TaskType) -> None:
        """
        Initializes the reward.

        :param task:         the environment in which the reward is used
        """
        self.__task = task

    def reset(self) -> None:
        """
        Resets the reward. Must be called at the beginning of each episode.
        """
        self._reset()
        self.__min_reward = self._get_min_reward_unnormalized()

    def _reset(self) -> None:
        """
        Resets the specific reward instance.
        """
        pass

    def calculate_reward(self, done) -> float:
        """
        Calculates the scaled reward for the current timestep.

        :return:            the reward for the current timestep
        """
        if self.__reward_condition is None or self.__reward_condition():
            reward = self._calculate_reward_unnormalized()
            if self.__clip:
                reward = max(reward, self.__min_reward)
            if self.__normalize:
                reward = reward / abs(self.__min_reward)
            reward_scale = self.__final_timestep_reward_scale if done else self.__intermediate_timestep_reward_scale
            return reward_scale * reward
        else:
            return 0.0

    @abstractmethod
    def _calculate_reward_unnormalized(self) -> float:
        """
        Calculates the reward (without scaling) for the current timestep. Must be overwritten by subclasses.
        """
        raise NotImplementedError("Method _calculate_reward_unnormalized must be overwritten.")

    @abstractmethod
    def _get_min_reward_unnormalized(self) -> float:
        """
        Returns the lowest possible (unnormalized) reward.

        :return:            the lowest possible (unnormalized) reward
        """
        raise NotImplementedError("Method _get_min_reward_unnormalized must be overwritten.")

    @property
    def task(self) -> Optional[TaskType]:
        """
        Returns the gym environment for which the reward is used.

        :return:            the gym environment for which the reward is used
        """
        return self.__task

    @property
    def name(self) -> str:
        """
        Returns the name of the reward.

        :return:            the name of the reward
        """
        return self.__name

    @property
    def name_abbreviation(self) -> str:
        """
        Returns the abbreviation of the name of the reward.

        :return:            the abbreviation of the name of the reward
        """
        return self.__name_abbreviation

    def __repr__(self):
        return "{}({}, {}, {})".format(type(self).__name__, self.__intermediate_timestep_reward_scale,
                                       self.__final_timestep_reward_scale, self.__clip)
