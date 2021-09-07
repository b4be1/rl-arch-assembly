from abc import ABC
from typing import Optional, List, Callable

import numpy as np

from .part_reward import PartReward


class PartMarkerVelocityReward(PartReward, ABC):
    def __init__(self, name_prefix: str, max_marker_velocity: float = 0.1,
                 intermediate_timestep_reward_scale: Optional[float] = 0.0,
                 final_timestep_reward_scale: Optional[float] = 0.25, abbreviated_name: Optional[str] = None,
                 reward_condition: Optional[Callable[[], bool]] = None):
        super().__init__(name_prefix + "_part_velocity_marker_reward", intermediate_timestep_reward_scale,
                         final_timestep_reward_scale, clip=True, abbreviated_name=abbreviated_name,
                         reward_condition=reward_condition)
        self.__last_marker_positions: List[np.ndarray] = []
        self.__max_marker_velocity = max_marker_velocity

    def _reset(self) -> None:
        parts = self._get_parts()
        self.__last_marker_positions = np.array([part.bounding_box_marker_positions for part in parts])

    def _calculate_reward_unnormalized(self) -> float:
        parts = self._get_parts()
        marker_positions = np.array([part.bounding_box_marker_positions for part in parts])
        marker_velocities_3d = (marker_positions - self.__last_marker_positions) / self.env.environment.time_step
        marker_velocities = np.linalg.norm(marker_velocities_3d, axis=-1)
        return -np.mean(marker_velocities)

    def _get_min_reward_unnormalized(self) -> float:
        return -self.__max_marker_velocity
