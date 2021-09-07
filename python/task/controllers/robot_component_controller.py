from abc import ABC
from typing import Optional, Literal

import gym

from assembly_gym.environment.generic import Robot, RobotComponent
from task import BaseTask
from task.controllers import Controller


class RobotComponentController(Controller, ABC):
    """
    Controls a robot component according to given actions (e.g. torques or target velocities).
    """

    def __init__(self, name_prefix: str, robot_name: str, robot_component: Literal["arm", "gripper"]):
        self.__name_prefix: str = name_prefix
        self.__robot_name: str = robot_name
        self.__robot_component_name: Literal["arm", "gripper"] = robot_component
        self.__robot: Optional[Robot] = None
        self.__robot_component: Optional[RobotComponent] = None
        super(RobotComponentController, self).__init__(
            "{}_controller_{}_{}".format(name_prefix, robot_name, robot_component))

    def initialize(self, task: BaseTask) -> gym.spaces.Space:
        """
        Initializes the controller with for a given task.
        """
        self.__robot = task.environment.robots[self.__robot_name]
        self.__robot_component = self.__robot.gripper if self.__robot_component_name == "gripper" else self.__robot.arm
        return super(RobotComponentController, self).initialize(task)

    @property
    def name_prefix(self) -> str:
        return self.__name_prefix

    @property
    def robot(self) -> Robot:
        return self.__robot

    @property
    def robot_component(self) -> Optional[RobotComponent]:
        return self.__robot_component

    @property
    def robot_component_name(self) -> Literal["arm", "gripper"]:
        return self.__robot_component_name
