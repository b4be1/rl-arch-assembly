from typing import List, Dict, Literal, Tuple

import numpy as np

from assembly_gym.environment.generic import JointMode
from task import BaseTask
from .robot_component_controller import RobotComponentController


class TorqueController(RobotComponentController):
    """
    A torque-based controller. An action consists of a torque for each joint.
    """

    def __init__(self, robot_name: str, robot_component: Literal["arm", "gripper"], torque_limits_lower: np.ndarray,
                 torque_limits_upper: np.ndarray):
        super().__init__("torque", robot_name, robot_component)
        self.__torque_limits_lower: np.ndarray = torque_limits_lower
        self.__torque_limits_upper: np.ndarray = torque_limits_upper

    def _initialize(self, task: BaseTask) -> Tuple[np.ndarray, np.ndarray]:
        self.robot_component.set_joint_mode(JointMode.TORQUE_CONTROL)
        return self.__torque_limits_lower, self.__torque_limits_upper

    def _actuate_denormalized(self, action: np.ndarray) -> None:
        assert self.robot_component.joint_mode == JointMode.TORQUE_CONTROL, \
            "The robot component is not in torque control mode"
        self.robot_component.set_joint_torques(action)

    @classmethod
    def from_parameters(cls, robot_name: str, robot_component: Literal["arm", "gripper"],
                        parameters: Dict[str, List[float]]) -> "TorqueController":
        """
        Create a TorqueController from a parameters dictionary.

        :param robot_name:                      the name of the robot that is controlled
        :param robot_component:                 whether the controller controls the arm or the gripper of the robot
        :param parameters:                      a dictionary containing the entries torque_limits_lower and
                                                torque_limits_upper
        :return:                                a TorqueController with the given parameters
        """
        torque_limits_lower = np.array(parameters["torque_limits_lower"])
        torque_limits_upper = np.array(parameters["torque_limits_upper"])
        return TorqueController(robot_name, robot_component, torque_limits_lower, torque_limits_upper)
