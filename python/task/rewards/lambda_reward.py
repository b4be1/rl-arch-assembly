from typing import TypeVar, Generic, Optional, Callable

from .reward import Reward
from task import BaseTask

TaskType = TypeVar("TaskType", bound=BaseTask)


class LambdaReward(Reward[TaskType], Generic[TaskType]):
    """
    A reward class that uses a custom function to calculate the rewards.
    """

    def __init__(self, name: str, lmbda: Callable[[TaskType], float], intermediate_timestep_reward_scale: float = 1.0,
                 final_timestep_reward_scale: Optional[float] = None, normalize: bool = True,
                 clip: bool = False, min_reward_unnormalized: float = -1,
                 abbreviated_name: Optional[str] = None):
        """
        :param name:                                the name of the reward (to be used as key in info dictionary
                                                    returned by the gym environment every step)
        :param lmbda:                               a function that takes the gym environment as input and produces a
                                                    reward for the current time step
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        :param normalize:                           whether the (unscaled) reward should be normalized to [-1, 0]
        :param clip:                                whether the (unnormalized, unscaled) reward should be clipped to not
                                                    go lower than min_reward_unnormalized
        :param min_reward_unnormalized:             the minimum possible reward (used for clipping and normalization;
                                                    ignored if clip and normalize are False)
        :param abbreviated_name:                    an abbreviation of the name of the reward that is displayed with the
                                                    value of the reward at the end of each episode (by the gym
                                                    environment)
        """
        super().__init__(name, intermediate_timestep_reward_scale, final_timestep_reward_scale, normalize, clip,
                         abbreviated_name)
        self.__lmbda = lmbda
        self.__min_reward_unnormalized = min_reward_unnormalized

    def _calculate_reward_unnormalized(self) -> float:
        return self.__lmbda(self.task)

    def _get_min_reward_unnormalized(self) -> float:
        return self.__min_reward_unnormalized
