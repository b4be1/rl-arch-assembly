from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, TYPE_CHECKING

import numpy as np

from .sensor import Sensor

if TYPE_CHECKING:
    from task import BaseTask

EnvType = TypeVar("EnvType", bound="BaseTask")
StateType = TypeVar("StateType")


class InvertibleSensor(Sensor[EnvType], Generic[EnvType, StateType], ABC):
    @abstractmethod
    def observation_to_state(self, observation: Dict[str, np.ndarray]) -> Dict[str, StateType]:
        pass
