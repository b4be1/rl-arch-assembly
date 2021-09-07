from abc import ABC, abstractmethod
from typing import TypeVar, Dict

import numpy as np

from .continuous_sensor import ContinuousSensor
from .invertible_sensor import InvertibleSensor

# TODO: Type definitions should probably be defined centrally somewhere
StateType = TypeVar("StateType")
EnvType = TypeVar("EnvType")


class InvertibleContinuousSensor(ContinuousSensor[EnvType], InvertibleSensor[EnvType, StateType], ABC):

    def observation_to_state(self, observation: Dict[str, np.ndarray]) -> Dict[str, StateType]:
        unnormalized_observation = self._undo_normalization_obs_dict(observation)
        return self._observation_to_state_unnormalized(unnormalized_observation)

    @abstractmethod
    def _observation_to_state_unnormalized(self, unnormalized_observation: Dict[str, np.ndarray]) -> \
            Dict[str, StateType]:
        pass

    # TODO: Need to disable clipping here (or ensure that rewards are clipped the same way)
    def _undo_normalization_obs_dict(self, normalized_observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {name: self._undo_normalization(normalized, self._limits[name][0], self._limits[name][1])
                for name, normalized in normalized_observation.items()}

    @staticmethod
    def _undo_normalization(normalized_observation: np.ndarray, limits_lower: np.ndarray, limits_upper: np.ndarray) \
            -> np.ndarray:
        obs_space_center = 0.5 * (limits_upper + limits_lower)
        obs_space_width = limits_upper - limits_lower
        return normalized_observation * (obs_space_width / 2) + obs_space_center
