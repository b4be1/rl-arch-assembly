from typing import Optional, Sequence

import numpy as np

from assembly_gym.util import Transformation
from .part_marker_sensor import PartMarkerSensor
from scene.part import Part


class ClosestOtherPartMarkerSensor(PartMarkerSensor):
    def __init__(self, robot_name: str = "ur10", nr_observed_parts: int = 5,
                 default_position: Optional[np.ndarray] = None):
        super().__init__(name_prefix="other_parts", nr_observed_parts=nr_observed_parts, use_target_markers=False,
                         default_position=default_position)
        self.__robot_name = robot_name

    def _get_observed_parts(self) -> Sequence[Part]:
        current_part = self.task.current_part
        placed_parts = self.task.placed_parts
        distances = [current_part.pyrep_shape.check_distance(placed_pyrep_part.pyrep_shape)
                     for placed_pyrep_part in placed_parts]
        indices_closest_parts = np.argsort(distances)[:self._nr_observed_parts]
        closest_parts = [placed_parts[idx] for idx in indices_closest_parts]

        return closest_parts

    def _observer_frame_pose(self) -> Optional[Transformation]:
        return self.task.environment.robots[self.__robot_name].arm.get_pose()
