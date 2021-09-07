from abc import ABC
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from .joint_sensor import JointSensor
from ..continuous_sensor import ContinuousSensor

if TYPE_CHECKING:
    from task import BaseTask


class JointAngleSensor(ContinuousSensor["BaseTask"], JointSensor, ABC):
    def __init__(self, joint_limits_tolerance: float = 0.1, **kwargs):
        """
        :param joint_limits_tolerance:          a fraction of the joint intervals that the joints might overshoot the
                                                joint limits (due to inaccuracies in the simulation)
        """
        super(JointAngleSensor, self).__init__(**kwargs)
        self.__joint_limits_tolerance = joint_limits_tolerance

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        joint_intervals = self.observed_robot_component.joint_intervals
        tolerance = (joint_intervals[:, 1] - joint_intervals[:, 0]) * self.__joint_limits_tolerance
        lower_limits = joint_intervals[:, 0] - tolerance
        upper_limits = joint_intervals[:, 1] + tolerance
        return {
            "{}_joint_pos".format(self.name_prefix): (lower_limits, upper_limits)
        }

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        pos = self.observed_robot_component.joint_positions
        return {
            "{}_joint_pos".format(self.name_prefix): pos
        }

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self._observe_unnormalized()
