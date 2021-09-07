from abc import ABC, abstractmethod
from typing import Optional, Callable

from assembly_gym.environment.generic import RobotComponent
from task import BaseTask
from ..reward import Reward


class RobotComponentReward(Reward[BaseTask], ABC):
    """
    An abstract base class for rewards that the agent receives for using its joints (e.g. costs for high torques).
    """

    def __init__(self, name_prefix: str, intermediate_timestep_reward_scale: float,
                 final_timestep_reward_scale: Optional[float], clip: bool = False,
                 abbreviated_name: Optional[str] = None,
                 reward_condition: Optional[Callable[[], bool]] = None):
        """
        :param name_prefix:                         the name prefix of the specific instance of the JointReward;
                                                    the complete reward name is name_prefix + "joint_reward"
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        :param clip:                                whether the (unscaled) reward should be clipped to not go lower
                                                    than -1
        :param abbreviated_name:                    an abbreviation of the name of the reward that is displayed with the
                                                    value of the reward at the end of each episode (by the gym
                                                    environment)
        :param reward_condition:                    a custom condition that specifies if the reward is active (i.e. the
                                                    reward is only calculated if reward_condition() is True, otherwise
                                                    the calculate_reward() returns 0)
        """
        name = name_prefix + "joint_reward"
        super().__init__(name, intermediate_timestep_reward_scale, final_timestep_reward_scale, clip,
                         abbreviated_name=abbreviated_name, reward_condition=reward_condition)
        self.__robot_component: Optional[RobotComponent] = None

    def _reset(self) -> None:
        self.__robot_component = self._get_robot_component()

    @abstractmethod
    def _get_robot_component(self) -> RobotComponent:
        """
        Extracts the robot component from the environment that should be taken into account during the reward
        calculation. Must be overwritten by subclasses.
        """
        raise NotImplementedError("Method _get_robot_component must be overwritten.")

    @property
    def robot_component(self) -> Optional[RobotComponent]:
        """
        Returns the robot component that should be taken into account during the reward calculation.

        :return:                the robot component that should be taken into account during the reward calculation
        """
        return self.__robot_component
