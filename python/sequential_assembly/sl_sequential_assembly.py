import time
from collections import Sequence
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet
from pybullet_planning import create_attachment, plan_joint_motion, BASE_LINK
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from aggregation import TemplatePart
from assembly_gym.environment.generic import JointMode
from assembly_gym.environment.pybullet import PybulletEnvironment

from gripper.schunk_gripper import SchunkGripper
from scene.part import Part
from scene.simulated_assembly_scene import SimulatedAssemblyScene
from .sequential_assembly_task import SequentialAssemblyTask
from assembly_gym.util import Transformation
from .util import gripper_pose_from_sl_pose


class PlanningFailedException(Exception):
    pass


class PlacingFailedException(Exception):
    pass


class SLSequentialAssembly:
    def __init__(self, sl_template_part: TemplatePart, scene_config_path: Path, placing_controller, # TODO: Type hint
                 calibrated_offsets_path: Optional[Path] = None, real_robot_ip: str = None):
        self.environment = PybulletEnvironment(headless=False, real_time_factor=1.0)
        self._sl_template_part = sl_template_part
        self.scene = SimulatedAssemblyScene([self._sl_template_part], scene_config_path,
                                            calibrated_offsets_path=calibrated_offsets_path,
                                            use_part_bounding_boxes=True)
        self._placing_controller = placing_controller
        self._real_robot_ip = real_robot_ip
        if real_robot_ip is not None:
            self._real_robot_interface_control = RTDEControlInterface(real_robot_ip)
            self._real_robot_interface_receive = RTDEReceiveInterface(real_robot_ip)
            self._real_gripper_interface = SchunkGripper()
        else:
            self._real_robot_interface_control = None
            self._real_robot_interface_receive = None
            self._real_gripper_interface = None
        self._parts = None
        self._logger = Logger("sl_sequential_assembly")

    def initialize(self):
        self.environment.initialize(0.05)
        self.scene.initialize_scene(self.environment)
        pybullet.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=160, cameraPitch=-40,
                                            cameraTargetPosition=[0, 0.4, 0.0])
        self.scene.robot.arm.set_joint_mode(JointMode.POSITION_CONTROL)
        self.scene.robot.gripper.set_joint_mode(JointMode.POSITION_CONTROL)
        self.environment.set_reset_checkpoint()
        self._placing_controller.initialize(self)
        self._reset()

    def _reset(self):
        # TODO: Should use reset checkpoints
        self.environment.reset_scene()
        self.scene.reset_scene()
        if self._real_robot_interface_control is not None:
            starting_conf = self._real_robot_interface_receive.getActualQ()
            self.scene.robot.arm.move_to_joint_positions(starting_conf)
            self.scene.robot.arm.set_joint_target_positions(starting_conf)
        else:
            self.scene.robot.place_gripper_at(Transformation.from_pos_euler([0, 0.85, 0.6], [np.pi, 0, 0]))
            self.scene.robot.arm.set_joint_target_positions(self.scene.robot.arm.joint_positions)

    def solve_task(self, construction_plan: SequentialAssemblyTask) -> None:
        self._parts = []
        for spawn_pose, target_pose in zip(construction_plan.spawn_poses, construction_plan.target_poses):
            part = Part(self._sl_template_part, target_pose, spawn_pose)
            self.scene.spawn_part(part)
            self._parts.append(part)
            part.scene_object.set_static(True)

        try:
            for part in self._parts:
                self.scene.update_target_marker(part)
                self._pick(part)
                self._transport(part)
                self._place(part)
            for _ in range(20):
                self.environment.step()
        finally:
            self._reset()

    def _plan_motion(self, target_ee_pose, grasped_part=None, min_distance_to_obstacles=0.005):
        reset_checkpoint = self.environment.physics_client.call(pybullet.saveState)  # TODO: This should be integrated into environment
        robot = self.scene.robot
        robot_pb = robot.arm.wrapped_body.unique_id
        joints = [j.wrapped_joint.joint_index for j in robot.arm.joints]
        target_joint_pos = np.array(self.scene.robot.solve_ik(target_ee_pose)[:6])
        if target_joint_pos[5] > 2 * np.pi:
            target_joint_pos[5] = target_joint_pos[5] - 2 * np.pi
        elif target_joint_pos[5] < -2 * np.pi:
            target_joint_pos[5] = target_joint_pos[5] + 2 * np.pi
        if grasped_part is not None:
            attachments = [create_attachment(robot.arm.wrapped_body.unique_id,
                                             robot.arm.wrapped_body.links["rh_p12_rn_r2"].link_index,
                                             # TODO: Should add an endeffector link
                                             grasped_part.scene_object.wrapped_body.unique_id)
                           ]
            extra_disabled_collisions = []
            for gripper_link in robot.gripper.wrapped_body.links.values():
                extra_disabled_collisions.append(((robot.gripper.wrapped_body.unique_id, gripper_link.link_index),
                                                 (grasped_part.scene_object.wrapped_body.unique_id, BASE_LINK)))
        else:
            attachments = []
            extra_disabled_collisions = []
        environment_obstacles = [self.scene.pickup_table.wrapped_body.unique_id,
                                 self.scene.place_table.wrapped_body.unique_id,
                                 self.scene.robot_socket.wrapped_body.unique_id,
                                 self.environment.ground_plane.unique_id]
        part_obstacles = [part.scene_object.wrapped_body.unique_id for part in self._parts if part != grasped_part]
        solution = plan_joint_motion(robot_pb, joints, target_joint_pos,
                                     obstacles=environment_obstacles + part_obstacles,
                                     attachments=attachments,
                                     extra_disabled_collisions=extra_disabled_collisions,
                                     max_distance=min_distance_to_obstacles,
                                     diagnosis=True)
        self.environment.physics_client.call(pybullet.restoreState, stateId=reset_checkpoint)
        if solution is None:
            raise PlanningFailedException("Planner failed to find a path from joint positions {} to {}"
                                          .format(tuple(robot.arm.joint_positions), tuple(target_joint_pos)))
        return solution

    def execute_plan(self, plan: Sequence[np.ndarray]) -> None:
        for configuration in plan:
            assert len(configuration) == 6, "Each configuration of the plan must have exactly 6 entries, got {}"\
                .format(len(configuration))
        for configuration in plan:
            self.scene.robot.arm.set_joint_target_positions(configuration)
            self.environment.step()
        for _ in range(10):
            self.environment.step()
        if self._real_robot_interface_control is not None:
            input("Press enter to execute this trajectory on the real robot")
            print("Executing...")
            velocity = 0.5
            acceleration = 0.7
            blend = 0.05
            plan_complete = [tuple(config) + (velocity, acceleration, blend) for config in plan[:-1]]
            plan_complete.append(tuple(plan[-1]) + (velocity, acceleration, 0.0))
            self._real_robot_interface_control.moveJ(plan_complete)
            reached_joint_pos = self._real_robot_interface_receive.getActualQ()
            abs_difference = np.sum(np.abs(np.array(reached_joint_pos) - np.array(plan[-1])))

            while abs_difference > 1e-2:
                self._logger.warning("Did not reach target joint positions. Target joint positions: {}, "
                                     "actual joint positions {}, difference: {}. Retrying..."
                                     .format(plan[-1], tuple(reached_joint_pos), abs_difference))
                self._real_robot_interface_control.moveJ(plan[-1], speed=velocity, acceleration=acceleration)
                reached_joint_pos = self._real_robot_interface_receive.getActualQ()
                abs_difference = np.sum(np.abs(np.array(reached_joint_pos) - np.array(plan[-1])))
            print("Done")

    def _pick(self, current_part: Part) -> None:
        pick_gripper_pose = gripper_pose_from_sl_pose(current_part.pose, self.scene.robot.gripper.pose)
        prepick_gripper_pose = Transformation(pick_gripper_pose.translation + np.array([0, 0, 0.05]),
                                              pick_gripper_pose.rotation)
        self.actuate_gripper(0.6)
        
        plan_prepick = self._plan_motion(prepick_gripper_pose, min_distance_to_obstacles=0.005)
        self.execute_plan(plan_prepick)

        plan_pick = self._plan_motion(pick_gripper_pose, min_distance_to_obstacles=0.002)
        self.execute_plan(plan_pick)

        self.actuate_gripper(0.8)
        current_part.scene_object.set_static(False)
        prepick_joint_angles = self.scene.robot.solve_ik(prepick_gripper_pose)[:6]     # TODO: This should be fixed in solve_ik
        self.execute_plan([prepick_joint_angles])

    def _transport(self, current_part: Part) -> None:
        transport_target_pose = self._get_transport_target_pose(current_part)
        gripper_target_pose = gripper_pose_from_sl_pose(transport_target_pose, self.scene.robot.gripper.pose)
        plan = self._plan_motion(gripper_target_pose, grasped_part=current_part)
        self.execute_plan(plan)

    def _place(self, current_part: Part):
        self._placing_controller.place(current_part)
        current_part.scene_object.set_static(True)

    def actuate_gripper(self, closure):
        self.scene.robot.gripper.set_joint_target_positions(np.array([closure]))
        for _ in range(10):  # TODO: Check whether gripper still moves (look at PyREPs actuate_gripper?)
            self.environment.step()
        if self._real_robot_interface_control is not None:
            if closure >= 0.75:
                input("Press enter to close gripper")
                self._real_gripper_interface.grip()
                time.sleep(2)
            else:
                input("Press enter to open gripper")
                self._real_gripper_interface.release()
                time.sleep(2)

    @staticmethod
    def _get_transport_target_pose(part: Part):
        return Transformation(part.target_pose.translation + np.array([0.0, 0.0, 0.1]), part.target_pose.rotation)
