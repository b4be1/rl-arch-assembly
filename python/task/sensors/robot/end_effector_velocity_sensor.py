import math
from typing import TYPE_CHECKING, Optional, Sequence

from ..object_velocity_sensor import ObjectVelocitySensor
from assembly_gym.environment.generic import Object

if TYPE_CHECKING:
    from task import BaseTask


class EndEffectorVelocitySensor(ObjectVelocitySensor["BaseTask"]):
    def __init__(self, robot_name: str = "ur10", linear_velocity_limit: float = 0.2,
                 angular_velocity_limit: float = math.pi / 4):
        super(EndEffectorVelocitySensor, self).__init__(
            ["end_effector"], linear_velocity_limit=linear_velocity_limit,
            angular_velocity_limit=angular_velocity_limit)
        self.__robot_name = robot_name

    def _get_observed_objects(self) -> Sequence[Object]:
        return [self.task.environment.robots[self.__robot_name].gripper]

    def _relative_to(self) -> Optional[Object]:
        return self.task.environment.robots[self.__robot_name].arm
