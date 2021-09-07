from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Sequence

from scene.part import Part
from task.sensors.sensor import Sensor

if TYPE_CHECKING:
    from task import StackingTask


class PartSensor(Sensor["StackingTask"], ABC):
    def __init__(self, name_prefix: str, nr_observed_parts: int, **kwargs):
        super().__init__(**kwargs)
        self.__name_prefix = name_prefix
        self._nr_observed_parts = nr_observed_parts

    @abstractmethod
    def _get_observed_parts(self) -> Sequence[Part]:
        raise NotImplementedError("Method _get_observed_parts must be overwritten")

    @property
    def name_prefix(self) -> str:
        return self.__name_prefix

    @property
    def nr_observed_parts(self) -> int:
        return self._nr_observed_parts
