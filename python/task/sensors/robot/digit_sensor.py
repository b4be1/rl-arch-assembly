from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from ..continuous_sensor import ContinuousSensor


if TYPE_CHECKING:
    from task import BaseTask

# TODO: repair this


class DigitSensor(ContinuousSensor["BaseTask"]):
    def __init__(self):
        super(DigitSensor, self).__init__()
        self.__digit_sensors = None

    def _initialize(self) -> None:
        super(DigitSensor, self)._initialize()
        self.__digit_sensors = [self.task.digit_sensors[k] for k in sorted(self.task.digit_sensors.keys())]
        assert len(self.__digit_sensors) > 0, "Digit sensors not found in the environment"
        resolutions = np.array([s.resolution for s in self.__digit_sensors])
        self.__resolution = resolutions[0]
        assert np.all(resolutions == self.__resolution), "All Digit sensors must have the same resolution"

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        obs_shape = (2, self.__resolution, 3)
        return {
            "digit_images": (np.zeros(obs_shape), np.ones(obs_shape))
        }

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self.observe()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        digit_images = np.array([s.get_measurements() for s in self.__digit_sensors])
        return {
            "digit_images": digit_images
        }
