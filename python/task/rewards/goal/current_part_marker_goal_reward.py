from typing import Dict, Any, Optional

import numpy as np

from task import StackingTask
from ..distances import ssd_log_distance
from ..goal.goal_reward import GoalReward
from assembly_gym.util import Transformation


# TODO: Sparse rewards should be handled in a more principled way --> Add a BinaryReward class?
# TODO: Avoid duplicated code
class CurrentPartMarkerGoalReward(GoalReward[StackingTask, Transformation]):
    def __init__(self, intermediate_timestep_reward_scale: float = 0.0,
                 final_timestep_reward_scale: Optional[float] = None, max_marker_distance: Optional[float] = None,
                 sparse: bool = False, distance_threshold_sparse: float = 0.01):
        super(CurrentPartMarkerGoalReward, self).__init__("part_marker_goal_reward", intermediate_timestep_reward_scale,
                                                          final_timestep_reward_scale, clip=True)
        self.__distance_measure = ssd_log_distance
        self.__sparse = sparse
        self.__distance_threshold_sparse = distance_threshold_sparse
        if max_marker_distance is None:
            self.__max_cost = 1.0
        else:
            self.__max_cost = self.__distance_measure(max_marker_distance ** 2)

    def _calculate_reward_unnormalized(self, achieved_goal: Dict[str, Transformation],
                                       desired_goal: Dict[str, Transformation], goal_rewards_info: Dict[str, Any]) \
            -> float:
        achieved_part_pose = achieved_goal["current_part"]
        desired_part_pose = desired_goal["current_part_target"]     # TODO: The naming should be fixed somehow
        relative_part_marker_positions = goal_rewards_info["current_base_part"].bounding_box_marker_positions
        achieved_marker_positions = achieved_part_pose.inv.transform(relative_part_marker_positions)
        desired_marker_positions = desired_part_pose.inv.transform(relative_part_marker_positions)
        differences = desired_marker_positions - achieved_marker_positions
        if self.__sparse:
            avg_euclidean_distance = np.mean(np.linalg.norm(differences, axis=-1))
            cost = self.__max_cost if avg_euclidean_distance > self.__distance_threshold_sparse else 0
        else:
            squared_dist = np.sum(differences ** 2, axis=-1)
            cost_per_marker = self.__distance_measure(squared_dist)

            # Average over all marker distances
            cost = np.mean(cost_per_marker)
        return -cost

    def information_to_store(self) -> Dict[str, Any]:
        return {"current_base_part": self.env.current_part.base_part}

    def _get_min_reward_unnormalized(self):
        return -self.__max_cost
