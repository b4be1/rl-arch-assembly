from typing import Sequence, Optional

from .part_marker_reward import PartMarkerReward
from scene.part import Part


class FuturePartMarkerReward(PartMarkerReward):
    """
    A concrete implementation of PartMarkerReward that takes all future parts (parts that are not yet in the scene) into
    account.
    """

    def __init__(self, max_marker_distance: Optional[float] = None,
                 intermediate_timestep_reward_scale: Optional[float] = 0.0,
                 final_timestep_reward_scale: Optional[float] = 0.25, logarithmic_penalty_weight: float = 0.01):
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
        """
        super().__init__("future", max_marker_distance, intermediate_timestep_reward_scale,
                         final_timestep_reward_scale, logarithmic_penalty_weight=logarithmic_penalty_weight)

    def _get_parts(self) -> Sequence[Part]:
        return self.task.future_parts
