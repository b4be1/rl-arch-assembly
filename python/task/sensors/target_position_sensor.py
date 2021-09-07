from typing import Dict, TYPE_CHECKING, Union, Tuple

import numpy as np

from .continuous_sensor import ContinuousSensor

if TYPE_CHECKING:
    from task import StabilizeTask


# TODO: This could probably be a pose sensor (that ignores the orientation)
class TargetPositionSensor(ContinuousSensor["StabilizeTask"]):
    def __init__(self, position_limit_lower: Union[float, np.ndarray] = -1,
                 position_limit_upper: Union[float, np.ndarray] = 1):
        self.__observation_name = "target_position"
        if np.isscalar(position_limit_lower):
            position_limit_lower = np.ones(3) * position_limit_lower
        if np.isscalar(position_limit_upper):
            position_limit_upper = np.ones(3) * position_limit_upper
        self.__limits = {self.__observation_name: (position_limit_lower, position_limit_upper)}
        super(TargetPositionSensor, self).__init__()

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return self.__limits

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self._observe_unnormalized()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        return {
            self.__observation_name: self.task.target_position_arm_frame
        }
