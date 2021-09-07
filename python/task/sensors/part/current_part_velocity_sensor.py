import math
from typing import Optional, Sequence

from assembly_gym.environment.generic import Object
from ..object_velocity_sensor import ObjectVelocitySensor
from task import StackingTask


class CurrentPartVelocitySensor(ObjectVelocitySensor[StackingTask]):
    def __init__(self, robot_name: str = "ur10", linear_velocity_limit: float = 1.0,
                 angular_velocity_limit: float = 2.0 * math.pi):
        super(CurrentPartVelocitySensor, self).__init__(
            ["current_part"], linear_velocity_limit=linear_velocity_limit,
            angular_velocity_limit=angular_velocity_limit)
        self.__robot_name = robot_name

    def _get_observed_objects(self) -> Sequence[Object]:
        return [self.task.current_part.scene_object]

    def _relative_to(self) -> Optional[Object]:
        return self.task.environment.robots[self.__robot_name].arm
