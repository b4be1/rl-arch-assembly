from typing import Sequence, Optional

import numpy as np

from .part_velocity_reward import PartVelocityReward
from scene.part import Part


class ReleasedPartVelocityReward(PartVelocityReward):
    """
    A concrete implementation of PartVelocityReward that takes all parts in the scene (placed parts, current part, and
    future parts) into account.
    """

    def __init__(self, robot_name: str = "ur10", max_linear_velocity: float = 1.0, max_angular_velocity: float = np.pi,
                 release_distance: float = 0.01, intermediate_timestep_reward_scale: Optional[float] = 0.01):
        """
        :param max_linear_velocity:                 the maximum linear velocity to use for normalizing the (unscaled)
                                                    reward to lie in [-1, 0]; the reward is clipped at -1 if the sum of
                                                    linear and angular velocity is higher than max_linear_velocity +
                                                    min_linear_velocity
        :param max_angular_velocity:                the maximum angular velocity to use for normalizing the (unscaled)
                                                    reward to lie in [-1, 0]
        :param release_distance:                    the minimum distance from the fingers to the part at which the part
                                                    is seen as released from the gripper
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        """
        super().__init__("released", max_linear_velocity, max_angular_velocity, intermediate_timestep_reward_scale,
                         None, abbreviated_name="released_part_vel", reward_condition=self._part_is_released)
        self.__release_distance = release_distance
        self.__robot_name = robot_name

    def _get_parts(self) -> Sequence[Part]:
        return [self.task.current_part]

    def _part_is_released(self) -> bool:
        return max(self.task.environment.robots[self.__robot_name].finger_distances_to_object(
            self.task.current_part.scene_object)) > self.__release_distance
