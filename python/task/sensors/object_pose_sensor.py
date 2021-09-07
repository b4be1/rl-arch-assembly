from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Generic, Sequence

from assembly_gym.environment.generic import Object
from assembly_gym.util import Transformation
from .pose_sensor import PoseSensor

TaskType = TypeVar("TaskType")


class ObjectPoseSensor(PoseSensor[TaskType], Generic[TaskType], ABC):
    def _observe_poses(self) -> Sequence[Transformation]:
        return [obj.get_pose() if obj is not None else None for obj in self._get_observed_objects()]

    def _observer_frame_pose(self) -> Optional[Transformation]:
        return self._relative_to().get_pose()

    @abstractmethod
    def _get_observed_objects(self) -> Sequence[Optional[Object]]:
        pass

    def _relative_to(self) -> Optional[Object]:
        return None
