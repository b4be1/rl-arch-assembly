from typing import Dict, List, Literal, Tuple
from typing import Optional

import numpy as np

from assembly_gym.environment.generic import JointMode
from task import BaseTask
from .robot_component_controller import RobotComponentController


class JointVelocityDifferenceController(RobotComponentController):
    """
    A controller based on joint velocities. An action consists of one target velocity difference for each joint.
    """

    def __init__(self, robot_name: str, robot_component: Literal["arm", "gripper"], acceleration_limits: np.ndarray,
                 vel_limits_lower: Optional[np.ndarray] = None, vel_limits_upper: Optional[np.ndarray] = None):
        """

        :param vel_limits_lower:    Lower velocity limit (rad/s)
        :param vel_limits_upper:    Upper velocity limit (rad/s)
        :param acceleration_limits: Acceleration limit (rad/s/step)
        """
        super().__init__("joint_velocity_difference", robot_name, robot_component)
        self.__vel_limits_lower: Optional[np.ndarray] = vel_limits_lower
        self.__vel_limits_upper: Optional[np.ndarray] = vel_limits_upper
        self.__acceleration_limits: np.ndarray = acceleration_limits

    def _initialize(self, task: BaseTask) -> Tuple[np.ndarray, np.ndarray]:
        self.robot_component.set_joint_mode(JointMode.VELOCITY_CONTROL)
        return -self.__acceleration_limits, self.__acceleration_limits

    def _actuate_denormalized(self, action: np.ndarray) -> None:
        assert self.robot_component.joint_mode == JointMode.VELOCITY_CONTROL, \
            "The robot component is not in velocity control mode"
        current_velocity = self.robot_component.get_joint_velocities()
        new_velocity = current_velocity + action
        new_velocity_clamped = np.minimum(np.maximum(new_velocity, self.__vel_limits_lower), self.__vel_limits_upper)
        self.robot_component.set_joint_target_velocities(new_velocity_clamped)

    @classmethod
    def from_parameters(cls, robot_name: str, robot_component: Literal["arm", "gripper"],
                        parameters: Dict[str, List[float]], time_step: float) \
            -> "JointVelocityDifferenceController":
        """
        Create a JointVelocityDifferenceController from a parameters dictionary.

        :param robot_name:                      the name of the robot that is controlled
        :param robot_component:                 whether the controller controls the arm or the gripper of the robot
        :param parameters:                      A dictionary that contains the entries joint_velocity_limits_lower
                                                and joint_velocity_limits_upper
        :param time_step:                       Time step the robot is controlled in.
        :return:                                A JointVelocityController with the given parameters
        """
        vel_limits_lower = np.array(parameters["joint_velocity_limits_lower"])
        vel_limits_upper = np.array(parameters["joint_velocity_limits_upper"])
        acc_limits = np.array(parameters["joint_acceleration_limits"]) * time_step
        return JointVelocityDifferenceController(robot_name, robot_component, vel_limits_lower, vel_limits_upper,
                                                 acc_limits)
