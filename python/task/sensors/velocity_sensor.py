import math
from abc import ABC, abstractmethod

from typing import Dict, TypeVar, Optional, Generic, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from .continuous_sensor import ContinuousSensor

TaskType = TypeVar("TaskType")


class VelocitySensor(ContinuousSensor[TaskType], Generic[TaskType], ABC):
    """
    A sensor for linear and angular velocities of multiple entities.
    """

    def __init__(self, name_prefixes: Sequence[str], linear_velocity_limit: Union[float, np.ndarray] = 1.0,
                 angular_velocity_limit: Union[float, np.ndarray] = 2.0 * math.pi):
        """
        :param name_prefixes:               one name prefix for each entity that is observed; the observation dict will
                                            then contain two entries for each name_prefix: name_prefix + "vel_angular"
                                            and name_prefix + "_vel_linear"
        :param linear_velocity_limit:       the maximum linear velocity along each axis; if only a single value is
                                            passed, that value is used for each axis
        :param angular_velocity_limit:      the maximum angular velocity around each axis; if only a single value is
                                            passed, that value is used for each axis
        """
        super(VelocitySensor, self).__init__()
        if np.isscalar(linear_velocity_limit):
            linear_velocity_limit = np.ones(3) * linear_velocity_limit
        if np.isscalar(angular_velocity_limit):
            angular_velocity_limit = np.ones(3) * angular_velocity_limit
        limits_angular = {"{}_vel_angular".format(prefix): (-angular_velocity_limit, angular_velocity_limit)
                          for prefix in name_prefixes}
        limits_linear = {"{}_vel_linear".format(prefix): (-linear_velocity_limit, linear_velocity_limit)
                         for prefix in name_prefixes}
        self.__limits = {**limits_angular, **limits_linear}
        self.__linear_velocity_limit = linear_velocity_limit
        self.__angular_velocity_limit = angular_velocity_limit
        self.__name_prefixes = tuple(name_prefixes)

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return self.__limits

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        # Assume that everything is stationary at the beginning of the episode (i.e. both linear and angular velocities
        # are 0).
        angular_obs = {
            "{}_vel_angular".format(prefix): np.zeros((3,))
            for prefix in self.__name_prefixes
        }
        linear_obs = {
            "{}_vel_linear".format(prefix): np.zeros((3,))
            for prefix in self.__name_prefixes
        }
        return {**angular_obs, **linear_obs}

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        # Read velocities and replace every None with 0 velocity
        velocities = [vel if vel is not None else (np.zeros(3), np.zeros(3)) for vel in self._observe_velocities()]
        lin, ang = zip(*velocities)

        observer_rotation = self._observer_frame_rotation()
        if observer_rotation is not None:
            lin = observer_rotation.apply(lin, inverse=True)

            # This works as we are using regular angular velocities (i.e. the rotation axis and the rotation velocity
            # that is represented by the axis' length)
            ang = observer_rotation.apply(ang, inverse=True)

        angular_obs = {
            "{}_vel_angular".format(prefix): a for a, prefix in zip(ang, self.__name_prefixes)
        }
        linear_obs = {
            "{}_vel_linear".format(prefix): l for l, prefix in zip(lin, self.__name_prefixes)
        }

        return {**angular_obs, **linear_obs}

    def _observer_frame_rotation(self) -> Optional[Rotation]:
        return None

    @abstractmethod
    def _observe_velocities(self) -> Sequence[Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Returns a sequence of velocities as (lin, ang) tuples, where lin is the linear velocity and ang the angular
        velocity (the axis of rotation scaled by the angular velocity).

        :return:
        """
        pass

    def __repr__(self):
        return "{}({}, {})".format(type(self).__name__, self.__linear_velocity_limit, self.__angular_velocity_limit)
