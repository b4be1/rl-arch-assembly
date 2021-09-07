from .placing_controller import PlacingController
from ..util import gripper_pose_from_sl_pose


class SimplePlacingController(PlacingController):
    def place(self, part):
        place_pose = gripper_pose_from_sl_pose(part.target_pose)
        initial_joint_pos = self._sl_sequential_assembly.scene.robot.arm.joint_positions
        target_joint_pos = self._sl_sequential_assembly.scene.robot.solve_ik(place_pose)[:6]
        self._sl_sequential_assembly.execute_plan([target_joint_pos])
        self._sl_sequential_assembly.actuate_gripper(0.5)
        self._sl_sequential_assembly.execute_plan([initial_joint_pos])

