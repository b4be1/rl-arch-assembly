import math
from abc import ABC
from typing import Optional, Callable

import numpy as np

from .. import distances
from .part_reward import PartReward


class PartVelocityReward(PartReward, ABC):
    """
    A reward that punishes (linear and angular) movement of parts.
    """

    def __init__(self, name_prefix: str, max_linear_velocity: float = 1.0,
                 max_angular_velocity: float = math.pi,
                 intermediate_timestep_reward_scale: Optional[float] = 0.0,
                 final_timestep_reward_scale: Optional[float] = 0.25, abbreviated_name: Optional[str] = None,
                 reward_condition: Optional[Callable[[], bool]] = None):
        """
        :param name_prefix:                         the name prefix of the specific instance of the PartVelocityReward;
                                                    the complete reward name is name_prefix + "part_velocity_reward"
        :param max_linear_velocity:                 the maximum linear velocity to use for normalizing the (unscaled)
                                                    reward to lie in [-1, 0]; the reward is clipped at -1 if the sum of
                                                    linear and angular velocity is higher than max_linear_velocity +
                                                    min_linear_velocity
        :param max_angular_velocity:                the maximum angular velocity to use for normalizing the (unscaled)
                                                    reward to lie in [-1, 0]
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        :param abbreviated_name:                    an abbreviation of the name of the reward that is displayed with the
                                                    value of the reward at the end of each episode (by the gym
                                                    environment)
        :param reward_condition:                    a custom condition that specifies if the reward is active (i.e. the
                                                    reward is only calculated if reward_condition() is True, otherwise
                                                    the calculate_reward() returns 0)
        """
        super().__init__(name_prefix + "_part_velocity_reward", intermediate_timestep_reward_scale,
                         final_timestep_reward_scale, clip=True, abbreviated_name=abbreviated_name,
                         reward_condition=reward_condition)
        self.__distance_measure = distances.ssd_log_distance
        self.__max_cost_per_part = max_linear_velocity + max_angular_velocity

    def _calculate_reward_unnormalized(self) -> float:
        velocities = np.array([part.scene_object.get_velocity() for part in self._get_parts()])
        linear_velocities = np.linalg.norm(velocities[:, 0], axis=-1)
        angular_velocities = np.linalg.norm(velocities[:, 1], axis=-1)
        average_linear_velocity = np.mean(linear_velocities)
        average_angular_velocity = np.mean(angular_velocities)
        cost = average_linear_velocity + average_angular_velocity
        return -cost

    def _get_min_reward_unnormalized(self) -> float:
        return -self.__max_cost_per_part * len(self._get_parts())
