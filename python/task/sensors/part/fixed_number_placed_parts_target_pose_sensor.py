from typing import Union, Optional, Sequence, TYPE_CHECKING

import numpy as np

from task.sensors.pose_sensor import PoseSensor
from assembly_gym.util import Transformation

if TYPE_CHECKING:
    from task import StackingTask


class FixedNumberPlacedPartsTargetPoseSensor(PoseSensor["StackingTask"]):
    """
    A sensor that provides the target poses of parts that are already placed to the agent. It assumes that the maximum
    number of placed parts is fixed and known when the sensor is constructed.  If the number of placed parts is greater
    than the number passed to the constructor, an exception is thrown.
    """

    def __init__(self, max_nr_placed_parts: int, robot_name: str = "ur10",
                 position_bounds_lower: Union[float, np.ndarray] = -2,
                 position_bounds_upper: Union[float, np.ndarray] = 2):
        """
        :param max_nr_placed_parts:             the number of parts that are already placed in the scene
        :param position_bounds_lower:
        :param position_bounds_upper:
        """
        names = ["placed_part_{}_target".format(i) for i in range(max_nr_placed_parts)]
        super(FixedNumberPlacedPartsTargetPoseSensor, self).__init__(
            names, position_bounds_lower=position_bounds_lower,
            position_bounds_upper=position_bounds_upper)
        self.__max_nr_placed_parts = max_nr_placed_parts
        self.__robot_name = robot_name

    def _observer_frame_pose(self) -> Optional[Transformation]:
        return self.task.environment.robots[self.__robot_name].arm.get_pose()

    def _observe_poses(self) -> Sequence[Optional[Transformation]]:
        target_poses = [part.target_pose for part in self.task.placed_parts]
        assert len(target_poses) <= self.__max_nr_placed_parts, \
            "Expected at maximum {} parts, but got {}".format(self.__max_nr_placed_parts, len(target_poses))
        # If the number of placed parts is smaller than the maximum number, pad with None
        return target_poses + [None] * (self.__max_nr_placed_parts - len(target_poses))
