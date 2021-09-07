from typing import Sequence

from .part_type_sensor import PartTypeSensor
from scene.part import Part


class CurrentPartTypeSensor(PartTypeSensor):
    def __init__(self):
        super().__init__(name_prefix="current_part", nr_observed_parts=1)

    def _get_observed_parts(self) -> Sequence[Part]:
        return [self.task.current_part]
