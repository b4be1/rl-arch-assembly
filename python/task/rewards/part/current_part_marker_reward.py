from typing import Sequence, Optional

from .part_marker_reward import PartMarkerReward
from scene.part import Part


class CurrentPartMarkerReward(PartMarkerReward):
    """
    A concrete implementation of PartMarkerReward that takes only the part that is currently placed by the robot into
    account.
    """

    def __init__(self, max_marker_distance: Optional[float] = None,
                 intermediate_timestep_reward_scale: Optional[float] = 0.0,
                 final_timestep_reward_scale: Optional[float] = 0.25, logarithmic_penalty_weight: float = 0.01,
                 marker_pos_tolerance: float = 0.0):
        """
        :param max_marker_distance:                 the maximum marker distance to use for normalizing the (unscaled)
                                                    reward to lie in [-1, 0]; the reward is clipped at -1 if the average
                                                    marker distance is higher than max_marker_distance
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        :param logarithmic_penalty_weight:          the weight of the logarithmic penalty
                                                    (see distances.ssd_log_distance)
        :param marker_pos_tolerance:                a distance to the target position at which a marker is seen as
                                                    correctly placed (at this distance the cost is 0)
        """
        super().__init__("current", max_marker_distance, intermediate_timestep_reward_scale,
                         final_timestep_reward_scale, logarithmic_penalty_weight=logarithmic_penalty_weight,
                         marker_pos_tolerance=marker_pos_tolerance)

    def _get_parts(self) -> Sequence[Part]:
        return [self.task.current_part]
