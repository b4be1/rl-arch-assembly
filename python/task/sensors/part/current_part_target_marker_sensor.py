from typing import Sequence, Optional

from assembly_gym.util import Transformation
from .part_marker_sensor import PartMarkerSensor
from scene.part import Part


class CurrentPartTargetMarkerSensor(PartMarkerSensor):
    def __init__(self):
        super().__init__(name_prefix="current_part_target", nr_observed_parts=1, use_target_markers=True)

    def _get_observed_parts(self) -> Sequence[Part]:
        return [self.task.current_part]

    def _observer_frame_pose(self) -> Optional[Transformation]:
        return self.task.environment.robot.arm.get_pose()
