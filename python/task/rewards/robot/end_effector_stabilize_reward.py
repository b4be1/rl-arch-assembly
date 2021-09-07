from typing import Optional

import numpy as np

from task import BaseTask
from task.rewards import Reward, distances


class EndEffectorStabilizeReward(Reward[BaseTask]):
    """
    A reward for stabilizing the end-effector at a given target position. To be used only with StabilizeEnv.
    """

    def __init__(self, robot_name: str = "ur10", max_pos_distance: float = 1.0,
                 intermediate_timestep_reward_scale: float = 0.8, final_timestep_reward_scale: Optional[float] = None):
        """
        :param max_pos_distance:                    the maximum distance to use for normalizing the (unscaled) reward to
                                                    lie in [-1, 0]
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        """
        self.__distance_measure = distances.ssd_log_distance
        name = "endeffector_stabilize_reward"
        super().__init__(name, intermediate_timestep_reward_scale, final_timestep_reward_scale, clip=False)
        self.__max_distance = self.__distance_measure(max_pos_distance ** 2)
        self.__robot_name = robot_name

    def _calculate_reward_unnormalized(self) -> float:
        distance = np.asscalar(np.sum(
            (self.task.environment.robots[self.__robot_name].gripper.pose.translation
             - self.task.target_position_world_frame) ** 2))
        cost = self.__distance_measure(distance)
        return -cost

    def _get_min_reward_unnormalized(self) -> float:
        return -self.__max_distance
