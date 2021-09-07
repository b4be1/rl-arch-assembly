import math
from typing import Optional, Sequence

from assembly_gym.environment.generic import Object
from ..object_velocity_sensor import ObjectVelocitySensor

from task import StackingTask


class FixedNumberPlacedPartsVelocitySensor(ObjectVelocitySensor[StackingTask]):
    """
    A sensor that provides the velocities of parts that are already placed to the agent. It assumes that the maximum
    number of placed parts is fixed and known when the sensor is constructed. If the number of placed parts is greater
    than the number passed to the constructor, an exception is thrown.
    """

    def __init__(self, max_nr_placed_parts: int, linear_velocity_limit: float = 0.2,
                 angular_velocity_limit: float = math.pi / 2):
        """
        :param max_nr_placed_parts:             the number of parts that are already placed in the scene
        :param linear_velocity_limit:           the maximum linear velocity limit used for normalization
        :param angular_velocity_limit:          the maximum angular velocity limit used for normalization
        """
        names = ["placed_part_{}".format(i) for i in range(max_nr_placed_parts)]
        super(FixedNumberPlacedPartsVelocitySensor, self).__init__(
            names, linear_velocity_limit=linear_velocity_limit,
            angular_velocity_limit=angular_velocity_limit)
        self.__max_nr_placed_parts = max_nr_placed_parts

    def _get_observed_objects(self) -> Sequence[Optional[Object]]:
        scene_objects = [part.scene_object for part in self.task.placed_parts]
        assert len(scene_objects) <= self.__max_nr_placed_parts, \
            "Expected at maximum {} parts, but got {}".format(self.__max_nr_placed_parts, len(scene_objects))
        # If the number of placed parts is smaller than the maximum number, pad with None
        return scene_objects + [None] * (self.__max_nr_placed_parts - len(scene_objects))

    def _relative_to(self) -> Optional[Object]:
        return self.task.robot.arm
