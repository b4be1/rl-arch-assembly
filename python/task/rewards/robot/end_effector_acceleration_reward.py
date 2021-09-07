from typing import Optional

import numpy as np

from task import BaseTask
from assembly_gym.environment.generic import RobotComponent
from task.rewards import Reward


class EndEffectorAccelerationReward(Reward[BaseTask]):
    """
    A reward for punishing high (linear and angular) end-effector accelerations.
    """

    def __init__(self, robot_name: str = "ur10", intermediate_timestep_reward_scale: float = 0.8,
                 final_timestep_reward_scale: Optional[float] = None, max_acceleration: float = 100.0):
        """
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        :param max_acceleration:                    the maximum acceleration to use for normalizing the (unscaled)
                                                    reward to lie in [-1, 0]
        """
        name = "endeffector_acceleration_reward"
        super().__init__(name, intermediate_timestep_reward_scale, final_timestep_reward_scale, clip=False,
                         abbreviated_name="ee_acc")
        self.__robot_name = robot_name
        self.__gripper: Optional[RobotComponent] = None
        self.__max_acceleration: float = max_acceleration

    def _reset(self) -> None:
        self.__gripper = self.task.environment.robots[self.__robot_name].gripper
        self.__previous_linear_velocity = np.zeros(3)
        self.__previous_angular_velocity = np.zeros(3)

    def _calculate_reward_unnormalized(self) -> float:
        linear_velocity, angular_velocity = self.__gripper.velocity
        linear_acceleration = (linear_velocity - self.__previous_linear_velocity) / self.task.environment.time_step
        angular_acceleration = (angular_velocity - self.__previous_angular_velocity) / self.task.environment.time_step
        linear_acceleration_len = np.linalg.norm(linear_acceleration)
        angular_acceleration_len = np.linalg.norm(angular_acceleration)
        cost = np.linalg.norm(linear_acceleration_len + angular_acceleration_len)
        return -cost

    def _get_min_reward_unnormalized(self) -> float:
        return -self.__max_acceleration
