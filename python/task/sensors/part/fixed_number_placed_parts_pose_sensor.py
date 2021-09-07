from typing import Sequence, Union, Optional, TYPE_CHECKING

import numpy as np

from assembly_gym.environment.generic import Object
from ..object_pose_sensor import ObjectPoseSensor

if TYPE_CHECKING:
    from task import StackingTask


class FixedNumberPlacedPartsPoseSensor(ObjectPoseSensor["StackingTask"]):
    """
    A sensor that provides the poses of parts that are already placed to the agent. It assumes that the maximum number
    of placed parts is fixed and known when the sensor is constructed. If the number of placed parts is greater than the
    number passed to the constructor, an exception is thrown.
    """

    def __init__(self, max_nr_placed_parts: int, robot_name: str = "ur10",
                 position_bounds_lower: Union[float, np.ndarray] = -2,
                 position_bounds_upper: Union[float, np.ndarray] = 2, angular_noise_variance: float = 0,
                 linear_noise_variance: float = 0):
        """
        :param max_nr_placed_parts:             the maximum number of parts that are already placed in the scene
        :param position_bounds_lower:
        :param position_bounds_upper:
        :param angular_noise_variance:
        :param linear_noise_variance:
        """
        names = ["placed_part_{}".format(i) for i in range(max_nr_placed_parts)]
        super(FixedNumberPlacedPartsPoseSensor, self).__init__(
            names, position_bounds_lower=position_bounds_lower, position_bounds_upper=position_bounds_upper,
            linear_noise_variance=linear_noise_variance, angular_noise_variance=angular_noise_variance)
        self.__max_nr_placed_parts = max_nr_placed_parts
        self.__robot_name = robot_name

    def _get_observed_objects(self) -> Sequence[Object]:
        parts = self.task.placed_parts
        assert len(parts) == self.__max_nr_placed_parts, "Expected {} placed parts but found {}.".format(
            self.__max_nr_placed_parts, len(parts))
        return [p.scene_object for p in parts]

    def _relative_to(self) -> Optional[Object]:
        return self.task.environment.robots[self.__robot_name].arm
