from typing import Optional, Sequence, Union, TYPE_CHECKING

import numpy as np

from task.sensors import ObjectPoseSensor
from assembly_gym.environment.generic import Object

if TYPE_CHECKING:
    from task import StackingTask


class CurrentPartPoseSensor(ObjectPoseSensor["StackingTask"]):
    # TODO: Its not really clear here why a robot name is needed
    def __init__(self, robot_name: str = "ur10", position_bounds_lower: Union[float, np.ndarray] = -2,
                 position_bounds_upper: Union[float, np.ndarray] = 2, angular_noise_variance: float = 0,
                 linear_noise_variance: float = 0):
        super(CurrentPartPoseSensor, self).__init__(
            ["current_part"], position_bounds_lower=position_bounds_lower, position_bounds_upper=position_bounds_upper,
            linear_noise_variance=linear_noise_variance, angular_noise_variance=angular_noise_variance)
        self.__robot_name = robot_name

    def _get_observed_objects(self) -> Sequence[Object]:
        return [self.task.current_part.scene_object]

    def _relative_to(self) -> Optional[Object]:
        return self.task.environment.robots[self.__robot_name].arm
