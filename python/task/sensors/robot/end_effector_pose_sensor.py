from typing import TYPE_CHECKING, Optional, Sequence

from ..object_pose_sensor import ObjectPoseSensor
from assembly_gym.environment.generic import Object

if TYPE_CHECKING:
    from task import BaseTask


class EndEffectorPoseSensor(ObjectPoseSensor["BaseTask"]):
    def __init__(self, robot_name: str = "ur10", angular_noise_variance: float = 0, linear_noise_variance: float = 0):
        super(EndEffectorPoseSensor, self).__init__(
            ["end_effector"], position_bounds_lower=-2, position_bounds_upper=2,
            angular_noise_variance=angular_noise_variance, linear_noise_variance=linear_noise_variance)
        self.__robot_name = robot_name

    def _get_observed_objects(self) -> Sequence[Object]:
        return [self.task.environment.robots[self.__robot_name].gripper]

    def _relative_to(self) -> Optional[Object]:
        return self.task.environment.robots[self.__robot_name].arm
