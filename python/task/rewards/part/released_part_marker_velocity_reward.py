from typing import Sequence, Optional


from .part_marker_velocity_reward import PartMarkerVelocityReward
from scene.part import Part


class ReleasedPartMarkerVelocityReward(PartMarkerVelocityReward):
    """
    A concrete implementation of PartVelocityReward that takes all parts in the scene (placed parts, current part, and
    future parts) into account.
    """

    def __init__(self, max_marker_velocity: float = 0.1, release_distance: float = 0.01,
                 intermediate_timestep_reward_scale: Optional[float] = 0.01):
        """
        :param max_marker_velocity:                 the maximum linear of a marker to use for normalizing the (unscaled)
                                                    reward to lie in [-1, 0]; the reward is clipped at -1 if the sum of
                                                    linear and angular velocity is higher than max_linear_velocity +
                                                    min_linear_velocity
        :param release_distance:                    the minimum distance from the fingers to the part at which the part
                                                    is seen as released from the gripper
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        """
        super().__init__("released", max_marker_velocity, intermediate_timestep_reward_scale,
                         None, abbreviated_name="released_part_vel", reward_condition=self._part_is_released)
        self.__release_distance = release_distance

    def _get_parts(self) -> Sequence[Part]:
        return [self.env.current_part]

    def _part_is_released(self) -> bool:
        return max(self.env.robot.finger_distances_to_object(self.env.current_part.scene_object)) \
               > self.__release_distance
