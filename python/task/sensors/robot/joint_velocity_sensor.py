from abc import ABC
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from .joint_sensor import JointSensor
from ..continuous_sensor import ContinuousSensor

if TYPE_CHECKING:
    from task import BaseTask


class JointVelocitySensor(ContinuousSensor["BaseTask"], JointSensor, ABC):
    def __init__(self, **kwargs):
        super(JointVelocitySensor, self).__init__(**kwargs)

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        # The joints often violate the velocity limits in simulation, thus the *2 to ensure that the agent observes the
        # velocity properly.
        upper_limits = self.observed_robot_component.upper_joint_velocity_limits * 2
        return {
            "{}_joint_vel".format(self.name_prefix): (-upper_limits, upper_limits)
        }

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        vel = self.observed_robot_component.joint_velocities
        return {
            "{}_joint_vel".format(self.name_prefix): vel
        }

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return {
            "{}_joint_vel".format(self.name_prefix): np.zeros((self.observed_robot_component.nr_joints,))
        }
