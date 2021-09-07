import functools
from abc import ABC
from typing import Optional, Callable

import numpy as np

from ..distances import ssd_log_distance
from .part_reward import PartReward


class PartMarkerReward(PartReward, ABC):
    """
    A reward that punishes the difference of the position of the 8 bounding box vertices to their corresponding target
    positions. The exact reward is the negative marker ssd_log_distance (see distances.py) averaged over the markers of
    all parts.
    """

    def __init__(self, name_prefix: str, max_marker_distance: Optional[float] = None,
                 intermediate_timestep_reward_scale: Optional[float] = 0.0,
                 final_timestep_reward_scale: Optional[float] = 0.25, abbreviated_name: Optional[str] = None,
                 logarithmic_penalty_weight: float = 0.01,
                 reward_condition: Optional[Callable[[], bool]] = None, marker_pos_tolerance: float = 0.0):
        """
        :param name_prefix:                         the name prefix of the specific instance of the PartMarkerReward;
                                                    the complete reward name is name_prefix + "part_marker_reward"
        :param max_marker_distance:                 the maximum marker distance to use for normalizing the (unscaled)
                                                    reward to lie in [-1, 0]; the reward is clipped at -1 if the average
                                                    marker distance is higher than max_marker_distance
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        :param abbreviated_name:                    an abbreviation of the name of the reward that is displayed with the
                                                    value of the reward at the end of each episode (by the gym
                                                    environment)
        :param logarithmic_penalty_weight:          the weight of the logarithmic penalty
                                                    (see distances.ssd_log_distance)
        :param reward_condition:                    a custom condition that specifies if the reward is active (i.e. the
                                                    reward is only calculated if reward_condition() is True, otherwise
                                                    the calculate_reward() returns 0)
        :param marker_pos_tolerance:                a distance to the target position at which a marker is seen as
                                                    correctly placed (at this distance the cost is 0)
        """
        self.__distance_measure = functools.partial(ssd_log_distance, logarithm_weight=logarithmic_penalty_weight)
        assert max_marker_distance > marker_pos_tolerance, \
            "The maximum marker distance must be larger than the positioning tolerance"
        self.__marker_pos_tolerance = marker_pos_tolerance
        if max_marker_distance is None:
            self.__max_cost = 1.0
        else:
            # The cost is zero as long as the part is in the tolerance of the target position. Therefore, the tolerance
            # needs to be subtracted from the max_marker_distance so that max_cost is the cost that is calculated if
            # the marker is at the given max_marker_distance.
            self.__max_cost = self.__distance_measure(max_marker_distance ** 2 - marker_pos_tolerance ** 2)
        super().__init__(name_prefix + "_part_marker_reward", intermediate_timestep_reward_scale,
                         final_timestep_reward_scale, clip=True, abbreviated_name=abbreviated_name,
                         reward_condition=reward_condition)

    def _calculate_reward_unnormalized(self) -> float:
        parts = self._get_parts()
        marker_positions_target = np.array([part.target_bounding_box_marker_positions for part in parts])
        marker_positions_actual = np.array([part.bounding_box_marker_positions for part in parts])

        # Calculate the ssd_log_distance for every marker position to its corresponding target position
        differences = marker_positions_target - marker_positions_actual
        dist = np.sum(differences ** 2, axis=-1)
        dist_tolerance = np.maximum(dist - np.ones_like(dist) * self.__marker_pos_tolerance ** 2, np.zeros_like(dist))
        cost_per_marker = self.__distance_measure(dist_tolerance)

        # Average over all marker distances
        cost = np.mean(cost_per_marker)
        return -cost

    def _get_min_reward_unnormalized(self) -> float:
        return -self.__max_cost
