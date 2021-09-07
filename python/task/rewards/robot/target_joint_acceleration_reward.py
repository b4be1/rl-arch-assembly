from abc import ABC
from typing import Optional

import numpy as np

from .robot_component_reward import RobotComponentReward


class TargetJointAccelerationReward(RobotComponentReward, ABC):
    """
    A reward that punishes high differences of the current joint velocities to the target joint velocities.
    """

    def __init__(self, name_prefix: str, max_joint_velocity_difference: float,
                 intermediate_timestep_reward_scale: float = 1e-4,
                 final_timestep_reward_scale: Optional[float] = None, exponent: float = 2.0):
        """
        :param name_prefix:                         the name prefix of the specific instance of the JointReward;
                                                    the complete reward name is name_prefix +
                                                    "target_velocity_difference_joint_reward"
        :param max_joint_velocity_difference:       the maximum joint velocity difference to use for normalizing the
                                                    (unscaled) reward to lie in [-1, 0]
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        :param exponent:                            the exponent to use in the calculation of the cost (e.g. 2 for a
                                                    squared cost)
        """
        super().__init__(name_prefix + "target_velocity_difference_", intermediate_timestep_reward_scale,
                         final_timestep_reward_scale, clip=False, abbreviated_name="arm_vel")
        self.__max_joint_velocity_difference: float = max_joint_velocity_difference
        self.__exponent: float = exponent

    def _get_min_reward_unnormalized(self):
        return self._distance_measure(np.ones(self.robot_component.nr_joints) * self.__max_joint_velocity_difference)

    def _calculate_reward_unnormalized(self) -> float:
        current_velocities = self.robot_component.joint_velocities
        current_target_velocities = self.robot_component.joint_target_velocities
        return -self._distance_measure(current_target_velocities - current_velocities)

    def _distance_measure(self, difference: np.ndarray) -> float:
        """
        A polynomial distance measure that is used to calculated the (unnormalized) cost from the differences in the
        joint velocities.

        :param difference:          the difference in the joint velocities
        :return:                    the (unnormalized) cost
        """
        return np.asscalar(np.sum(np.abs(difference) ** self.__exponent))
