from typing import Union, Optional, Sequence, TYPE_CHECKING

import numpy as np

from ..pose_sensor import PoseSensor
from assembly_gym.util import Transformation

if TYPE_CHECKING:
    from task.stacking_task import StackingTask


class CurrentPartTargetPoseSensor(PoseSensor["StackingTask"]):
    def __init__(self, robot_name: str = "ur10", position_bounds_lower: Union[float, np.ndarray] = -2,
                 position_bounds_upper: Union[float, np.ndarray] = 2):
        super(CurrentPartTargetPoseSensor, self).__init__(
            ["current_part_target"], position_bounds_lower=position_bounds_lower,
            position_bounds_upper=position_bounds_upper)
        self.__robot_name = robot_name

    def _observer_frame_pose(self) -> Optional[Transformation]:
        return self.task.environment.robots[self.__robot_name].arm.get_pose()

    def _observe_poses(self) -> Sequence[Transformation]:
        return [self.task.current_part.target_pose]

