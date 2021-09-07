from abc import ABC
from typing import Dict, Optional, Union, Tuple

import numpy as np

from assembly_gym.util import Transformation
from .part_sensor import PartSensor
from ..continuous_sensor import ContinuousSensor
from task import StackingTask


class PartMarkerSensor(ContinuousSensor[StackingTask], PartSensor, ABC):
    def __init__(self, use_target_markers: bool, default_position: Optional[np.array] = None,
                 lower_position_limits: Union[float, np.ndarray] = -2,
                 upper_position_limits: Union[float, np.ndarray] = 2, **kwargs):
        super(PartMarkerSensor, self).__init__(**kwargs)
        if np.isscalar(lower_position_limits):
            lower_position_limits = np.ones((3, 1)) * lower_position_limits
        if np.isscalar(upper_position_limits):
            upper_position_limits = np.ones((3, 1)) * upper_position_limits
        lower_position_limits_tiled = np.tile(lower_position_limits, (self.nr_observed_parts, 1, 8))
        upper_position_limits_tiled = np.tile(upper_position_limits, (self.nr_observed_parts, 1, 8))

        if default_position is None:
            default_position = lower_position_limits
        self.__default_position = default_position.reshape((1, 3, 1))
        if use_target_markers:
            self.__obs_space_name = "{}_target_marker_pos".format(self.name_prefix)
        else:
            self.__obs_space_name = "{}_marker_pos".format(self.name_prefix)

        self.__limits = {self.__obs_space_name: (lower_position_limits_tiled, upper_position_limits_tiled)}

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return self.__limits

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self._observe_unnormalized()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        parts = self._get_observed_parts()
        markers = np.array([part.bounding_box_marker_positions for part in parts])
        observer_frame_pose = self._observer_frame_pose()
        if observer_frame_pose is not None:
            markers = observer_frame_pose.transform(markers, inverse=True)

        if len(parts) < self._nr_observed_parts:
            fill_positions = np.tile(self.__default_position, (self._nr_observed_parts - len(parts), 1, 8))
            if len(parts) == 0:
                markers = fill_positions
            else:
                markers = np.concatenate((markers, fill_positions), axis=0)
        return {
            self.__obs_space_name: markers
        }

    def _observer_frame_pose(self) -> Optional[Transformation]:
        return None
