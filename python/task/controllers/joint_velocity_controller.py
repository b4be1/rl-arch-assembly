from typing import Dict, List, Tuple, Literal
from typing import Optional

import numpy as np

from assembly_gym.environment.generic import JointMode
from task import BaseTask
from .robot_component_controller import RobotComponentController


class JointVelocityController(RobotComponentController):
    """
    A controller based on joint velocities. An action consists of one target velocity for each joint.
    """

    def __init__(self, robot_name: str, robot_component: Literal["arm", "gripper"],
                 action_limits_lower: Optional[np.ndarray] = None, action_limits_upper: Optional[np.ndarray] = None):
        super().__init__("joint_velocity", robot_name, robot_component)
        assert (action_limits_lower is None) == (action_limits_upper is None)
        self.__action_limits_lower: Optional[np.ndarray] = action_limits_lower
        self.__action_limits_upper: Optional[np.ndarray] = action_limits_upper

    def _initialize(self, task: BaseTask) -> Tuple[np.ndarray, np.ndarray]:
        self.robot_component.set_joint_mode(JointMode.VELOCITY_CONTROL)
        if self.__action_limits_lower is None or self.__action_limits_upper is None:
            self.__action_limits_lower = -self.robot_component.upper_joint_velocity_limits
            self.__action_limits_upper = self.robot_component.upper_joint_velocity_limits
        return self.__action_limits_lower, self.__action_limits_upper

    def _actuate_denormalized(self, action: np.ndarray) -> None:
        assert self.robot_component.joint_mode == JointMode.VELOCITY_CONTROL, \
            "The robot component is not in velocity control mode"
        self.robot_component.set_joint_target_velocities(action)

    @classmethod
    def from_parameters(cls, robot_name: str, robot_component: Literal["arm", "gripper"],
                        parameters: Dict[str, List[float]]) -> "JointVelocityController":
        """
        Create a JointVelocityController from a parameters dictionary.

        :param robot_name:                      the name of the robot that is controlled
        :param robot_component:                 whether the controller controls the arm or the gripper of the robot
        :param parameters:                      a dictionary that contains the entries joint_velocity_limits_lower
                                                and joint_velocity_limits_upper
        :return:                                a JointVelocityController with the given parameters
        """
        vel_limits_lower = np.array(parameters["joint_velocity_limits_lower"])
        vel_limits_upper = np.array(parameters["joint_velocity_limits_upper"])
        return JointVelocityController(robot_name, robot_component, vel_limits_lower, vel_limits_upper)
