from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Generic, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from assembly_gym.environment.generic import Object
from .velocity_sensor import VelocitySensor

if TYPE_CHECKING:
    from task import BaseTask

TaskType = TypeVar("TaskType", bound="BaseTask")


class ObjectVelocitySensor(VelocitySensor[TaskType], Generic[TaskType], ABC):
    def _observe_velocities(self) -> Sequence[Optional[Tuple[np.ndarray, np.ndarray]]]:
        return [obj.get_velocity() if obj is not None else None for obj in self._get_observed_objects()]

    def _observer_frame_rotation(self) -> Optional[Rotation]:
        return Rotation.from_quat(self._relative_to().get_pose().quaternion)

    @abstractmethod
    def _get_observed_objects(self) -> Sequence[Object]:
        pass

    def _relative_to(self) -> Optional[Object]:
        return None
