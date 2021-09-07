from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from ..continuous_sensor import ContinuousSensor

if TYPE_CHECKING:
    from task import StackingTask


# TODO: This should be a DiscreteSensor
class CurrentPartReleasedSensor(ContinuousSensor["StackingEnv"]):
    def __init__(self, part_release_distance: float = 0.05, **kwargs):
        super().__init__(normalize=False, clip=False, **kwargs)
        self.__part_release_distance = part_release_distance
        self.__observation_name = "current_part_released"

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {self.__observation_name: (np.zeros(1), np.ones(1))}

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self._observe_unnormalized()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        part = self.task.current_part
        robot = self.task.robot
        # TODO: Fix typing
        part_released = max(robot.finger_distances_to_object(part.scene_object)) \
                        > self.__part_release_distance
        return {self.__observation_name: np.array([float(part_released)])}
