from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from .robot_component_controller import RobotComponentController
from assembly_gym.environment.generic import JointMode


class JointPositionController(RobotComponentController):
    """
    A controller based on joint positions. An action consists of one target position for each joint. A PID controller
    is then used (inside the simulation) to execute the action.
    """

    def __init__(self, robot_name: str, robot_component: Literal["arm", "gripper"],
                 action_limits_lower: Optional[np.ndarray] = None, action_limits_upper: Optional[np.ndarray] = None):
        super().__init__("joint_position", robot_name, robot_component)
        assert (action_limits_lower is None) == (action_limits_upper is None)
        self.__action_limits_lower: Optional[np.ndarray] = action_limits_lower
        self.__action_limits_upper: Optional[np.ndarray] = action_limits_upper

    def _initialize(self, task) -> Tuple[np.ndarray, np.ndarray]:
        self.robot_component.set_joint_mode(JointMode.POSITION_CONTROL)
        if self.__action_limits_lower is None or self.__action_limits_upper is None:
            self.__action_limits_lower = self.robot_component.joint_intervals[0]
            self.__action_limits_upper = self.robot_component.joint_intervals[1]
        return self.__action_limits_lower, self.__action_limits_upper

    def _actuate_denormalized(self, action: np.ndarray) -> None:
        assert self.robot_component.joint_mode == JointMode.POSITION_CONTROL, \
            "The robot component is not in position control mode"
        self.robot_component.set_joint_target_positions(action)

    @classmethod
    def from_parameters(cls, robot_name: str, robot_component: Literal["arm", "gripper"],
                        parameters: Dict[str, List[float]]) -> "JointPositionController":
        """
        Create a JointPositionController from a parameters dictionary.

        :param robot_name:                      the name of the robot that is controlled
        :param robot_component:                 whether the controller controls the arm or the gripper of the robot
        :param parameters:                      a dictionary that contains the entries joint_position_limits_lower
                                                and joint_position_limits_upper
        :return:                                a JointPositionController with the given parameters
        """
        vel_limits_lower = np.array(parameters["joint_position_limits_lower"])
        vel_limits_upper = np.array(parameters["joint_position_limits_upper"])
        return JointPositionController(robot_name, robot_component, vel_limits_lower, vel_limits_upper)
