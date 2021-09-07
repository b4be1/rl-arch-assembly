from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from .continuous_sensor import ContinuousSensor

if TYPE_CHECKING:
    from task import BaseTask


class TimeSensor(ContinuousSensor["BaseTask"]):
    """
    A sensor that provides the current step in the gym environment as observation to the agent. As described in "Time
    Limits in Reinforcement Learning" by Pardo et al. this observation is necessary to avoid violating the Markov
    property for the environment.
    """

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {"time": (np.zeros(1), np.array([self.task.time_limit_steps]))}

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self._observe_unnormalized()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        return {"time": np.array([self.task.current_step])}
