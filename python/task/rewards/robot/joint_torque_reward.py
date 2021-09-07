from abc import ABC
from typing import Optional

from .robot_component_reward import RobotComponentReward


class JointTorqueReward(RobotComponentReward, ABC):
    """
    A reward that punishes high joint torques.
    """

    def __init__(self, name_prefix: str, max_torque: float, intermediate_timestep_reward_scale: float = 1e-4,
                 final_timestep_reward_scale: Optional[float] = None, exponent: float = 2.0):
        """
        :param name_prefix:                         the name prefix of the specific instance of the JointReward;
                                                    the complete reward name is name_prefix + "torque_joint_reward"
        :param max_torque:                          the maximum torque to use for normalizing the (unscaled) reward to
                                                    lie in [-1, 0]
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        :param exponent:                            the exponent to use in the calculation of the cost (e.g. 2 for a
                                                    squared cost)
        """
        super().__init__(name_prefix + "torque_", intermediate_timestep_reward_scale, final_timestep_reward_scale,
                         clip=False)
        self.__exponent: float = exponent
        self.__max_torque: float = max_torque

    def _get_min_reward_unnormalized(self) -> float:
        return sum([self.__max_torque ** self.__exponent for _ in self.robot_component])

    def _calculate_reward_unnormalized(self) -> float:
        return sum(joint.get_torque() ** self.__exponent for joint in self.robot_component)
