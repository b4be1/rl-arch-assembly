from typing import Tuple

import numpy as np

from .controller import Controller
from task import BaseTask


class ConstantActionController(Controller):
    """
    Provides a specified constant action value to the underlying controller
    """

    def __init__(self, inner_controller: Controller, constant_action: np.ndarray):
        super().__init__("const_{}".format(inner_controller.name))
        self._inner_controller: Controller = inner_controller
        self._constant_action: np.ndarray = constant_action

    def _actuate_denormalized(self, action: np.ndarray) -> None:
        self._inner_controller._actuate_denormalized(self._constant_action)

    def _initialize(self, task: BaseTask) -> Tuple[np.ndarray, np.ndarray]:
        inner_controller_action_space = self._inner_controller.initialize(task)
        assert self._constant_action.shape == inner_controller_action_space.shape
        return np.array([]), np.array([])

    def __repr__(self):
        return "ConstantActionController for {}".format(self._inner_controller)
