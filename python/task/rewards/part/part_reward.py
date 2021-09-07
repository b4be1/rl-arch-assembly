from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Sequence, Optional, Callable

from ..reward import Reward
from scene.part import Part

if TYPE_CHECKING:
    from task import StackingTask


class PartReward(Reward["StackingTask"], ABC):
    """
    An abstract base class for all rewards that depend on parts in the scene.
    """

    def __init__(self, name: str, intermediate_timestep_reward_scale: Optional[float],
                 final_timestep_reward_scale: Optional[float], clip: bool = False,
                 abbreviated_name: Optional[str] = None,
                 reward_condition: Optional[Callable[[], bool]] = None):
        """
        :param name:                                the name of the reward (to be used as key in info dictionary
                                                    returned by the gym environment every step)
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        :param clip:                                whether the reward should be clipped to not go lower than -1
        :param abbreviated_name:                    an abbreviation of the name of the reward that is displayed with the
                                                    value of the reward at the end of each episode (by the gym
                                                    environment)
        :param reward_condition:                    a custom condition that specifies if the reward is active (i.e. the
                                                    reward is only calculated if reward_condition() is True, otherwise
                                                    the calculate_reward() returns 0)
        """
        super().__init__(name, intermediate_timestep_reward_scale, final_timestep_reward_scale, clip=clip,
                         abbreviated_name=abbreviated_name, reward_condition=reward_condition)
        self.__parts: Optional[Sequence[Part]] = None

    def _reset(self) -> None:
        return

    @abstractmethod
    def _get_parts(self) -> Sequence[Part]:
        """
        Extracts the parts from the environment that should be taken into account during the reward calculation.
        Must be overwritten by subclasses.
        """
        raise NotImplementedError("Method _get_parts must be overwritten")
