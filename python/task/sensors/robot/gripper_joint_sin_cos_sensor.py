from assembly_gym.environment.generic import RobotComponent
from .joint_sin_cos_sensor import JointSinCosSensor


class GripperJointSinCosSensor(JointSinCosSensor):
    def __init__(self, robot_name: str = "ur10"):
        super(GripperJointSinCosSensor, self).__init__(name_prefix="gripper")
        self.__robot_name = robot_name

    def _get_observed_robot_component(self) -> RobotComponent:
        return self.task.environment.robots[self.__robot_name].gripper
