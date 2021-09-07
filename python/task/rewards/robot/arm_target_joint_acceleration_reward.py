from typing import Optional

from assembly_gym.environment.generic import RobotComponent
from .target_joint_acceleration_reward import TargetJointAccelerationReward


class ArmTargetJointAccelerationReward(TargetJointAccelerationReward):
    """
    A reward that punishes high differences of the current joint velocities to the target joint velocities for the arm.
    """

    def __init__(self, robot_name: str = "ur10", max_joint_velocity_difference: float = 2.0,
                 intermediate_timestep_reward_scale: float = 1e-4,
                 final_timestep_reward_scale: Optional[float] = None, exponent: float = 2.0):
        """
        :param max_joint_velocity_difference:       the maximum joint velocity difference to use for normalizing the
                                                    (unscaled) reward to lie in [-1, 0]
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        :param final_timestep_reward_scale:         scaling factor (applied to the reward at the step in which the gym
                                                    environment terminates)
        :param exponent:                            the exponent to use in the calculation of the cost (e.g. 2 for a
                                                    squared cost)
        """
        super().__init__("arm_", max_joint_velocity_difference, intermediate_timestep_reward_scale,
                         final_timestep_reward_scale, exponent)
        self.__robot_name = robot_name

    def _get_robot_component(self) -> RobotComponent:
        return self.task.environment.robots[self.__robot_name].arm
