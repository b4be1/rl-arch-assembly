from typing import Sequence, Optional

import numpy as np

from .part_velocity_reward import PartVelocityReward
from scene.part import Part


class CurrentPartVelocityReward(PartVelocityReward):
    """
    A concrete implementation of PartVelocityReward that takes all parts in the scene (placed parts, current part, and
    future parts) into account.
    """

    def __init__(self, max_linear_velocity: float = 1.0,
                 max_angular_velocity: float = np.pi,
                 intermediate_timestep_reward_scale: Optional[float] = 0.0,
                 final_timestep_reward_scale: Optional[float] = 0.3):
        """
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
        """
        super().__init__("current", max_linear_velocity, max_angular_velocity, intermediate_timestep_reward_scale,
                         final_timestep_reward_scale, abbreviated_name="current_part_vel")

    def _get_parts(self) -> Sequence[Part]:
        return [self.task.current_part]
