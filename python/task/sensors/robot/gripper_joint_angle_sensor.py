from assembly_gym.environment.generic import RobotComponent
from .joint_angle_sensor import JointAngleSensor


class GripperJointAngleSensor(JointAngleSensor):
    def __init__(self, robot_name: str = "ur10", joint_limits_tolerance: float = 0.0):
        super(GripperJointAngleSensor, self).__init__(name_prefix="gripper",
                                                      joint_limits_tolerance=joint_limits_tolerance)
        self.__robot_name = robot_name

    def _get_observed_robot_component(self) -> RobotComponent:
        return self.task.environment.robots[self.__robot_name].gripper
