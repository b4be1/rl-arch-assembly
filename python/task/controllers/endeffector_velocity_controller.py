from typing import Dict, List, Union, Tuple

import numpy as np

from assembly_gym.environment.generic import JointMode
from task import BaseTask
from .robot_component_controller import RobotComponentController


class EndEffectorVelocityController(RobotComponentController):
    """
    An end-effector velocity-based arm controller. An action consists of a (cartesian) target velocity for the
    end-effector. If a target velocity is not possible (e.g. because of singularities), the closest possible velocity
    (according to the L2 norm) velocity is used).
    """

    def __init__(self, robot_name: str, linear_limits_lower: Union[np.ndarray, float],
                 linear_limits_upper: Union[np.ndarray, float], angular_limits_lower: Union[np.ndarray, float],
                 angular_limits_upper: Union[np.ndarray, float], smoothness_penalty_weight: float = 0.1):
        super().__init__("end_effector_velocity", robot_name, "arm")
        self.__linear_limits_lower: np.ndarray = linear_limits_lower
        self.__linear_limits_upper: np.ndarray = linear_limits_upper
        self.__angular_limits_lower: np.ndarray = angular_limits_lower
        self.__angular_limits_upper: np.ndarray = angular_limits_upper
        self._smoothness_penalty_weight: float = smoothness_penalty_weight

    def _actuate_denormalized(self, action: np.ndarray) -> None:
        rc = self.robot_component
        assert rc is self.robot.arm, "An end-effector velocity controller can only be used for arms"
        current_velocity = np.array(rc.joint_velocities)
        right_side = np.concatenate([action, self._smoothness_penalty_weight * current_velocity])
        jacobian = self.robot.get_end_effector_jacobian()
        mat = np.concatenate([jacobian, self._smoothness_penalty_weight * np.eye(current_velocity.shape[0])])
        joint_target_velocities = np.linalg.lstsq(mat, right_side, rcond=None)[0]
        joint_limits = rc.upper_joint_velocity_limits
        relative_velocities = joint_target_velocities / joint_limits
        scaling = 1 / np.maximum(np.max(np.abs(relative_velocities)), 1)
        scaled_velocities = joint_target_velocities * scaling
        rc.set_joint_target_velocities(scaled_velocities)

    def _initialize(self, task: BaseTask) -> Tuple[np.ndarray, np.ndarray]:
        self.robot_component.set_joint_mode(JointMode.VELOCITY_CONTROL)
        limits_lower = np.concatenate((self.__linear_limits_lower * np.ones(3),
                                       self.__angular_limits_lower * np.ones(3)))
        limits_upper = np.concatenate((self.__linear_limits_upper * np.ones(3),
                                       self.__angular_limits_upper * np.ones(3)))
        return limits_lower, limits_upper

    @classmethod
    def from_parameters(cls, robot_name: str, parameters: Dict[str, List[float]],
                        smoothness_penalty_weight: float = 0.1) -> "EndEffectorVelocityController":
        """
        Create an EndEffectorVelocityController from an parameters dictionary.

        :param robot_name:                      the name of the robot that is controlled
        :param parameters:                      a dictionary containing the entries end_effector_velocity_limits_lower
                                                and end_effector_velocity_limits_upper
        :param smoothness_penalty_weight:       TODO
        :return:                                an EndEffectorVelocityController with the given parameters
        """
        kwargs = {
            "{}_limits_{}".format(t, l): parameters["end_effector_{}_velocity_limits_{}".format(t, l)]
            for t in ["linear", "angular"]
            for l in ["upper", "lower"]
        }
        return EndEffectorVelocityController(robot_name, **kwargs, smoothness_penalty_weight=smoothness_penalty_weight)
