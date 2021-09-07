import argparse
import json
import math
from pathlib import Path
from time import sleep

import pybullet
import numpy as np


from assembly_gym.environment.generic import JointMode
from assembly_gym.environment.pybullet import PybulletEnvironment
from assembly_gym.environment.simulation import SimulationRobot, SimulationEnvironment, SimulationRobotComponent
from scene.part import Part
from aggregation import AggregationPart, AggregationProperties
from scene.simulated_assembly_scene import SimulatedAssemblyScene
from assembly_gym.util import Transformation


def move_arm(env: SimulationEnvironment, robot: SimulationRobot, ee_pose: Transformation, nr_steps: int = 25):
    start_positions = np.array(robot.arm.joint_positions)
    target_positions = np.array(robot.solve_ik(ee_pose)[:-2])
    for i in range(nr_steps):
        curr_target = start_positions * (nr_steps - i) / nr_steps + target_positions * i / nr_steps
        robot.arm.set_joint_target_positions(curr_target)
        env.step()
    for _ in range(2):
        env.step()


def actuate_gripper(env: SimulationEnvironment, gripper: SimulationRobotComponent, gripper_position: float):
    gripper.set_joint_target_positions(np.array([gripper_position]))
    for _ in range(10):
        env.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Uses the robot to build a small aggregation in simulation.")
    parser.add_argument("tasks", type=str, help="The path to the task definitions.")
    parser.add_argument("properties", type=str, help="The path to the properties.")
    parser.add_argument("-i", "--intelligent", action="store_true", help="Whether to make the controller intelligent.")
    args = parser.parse_args()
    tasks_path = Path(args.tasks)

    env = PybulletEnvironment(headless=False, real_time_factor=1.0)

    try:
        env.initialize(0.05)
        pybullet.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=180, cameraPitch=-30,
                                            cameraTargetPosition=[0, 0.2, 0.6])

        aggregations_path = Path(args.tasks)
        properties_path = Path(args.properties)

        properties = AggregationProperties.load(properties_path)
        global_target_shape = properties.global_target_shape

        with (tasks_path / "0000.json").open() as f:
            current_task = json.load(f)

        # Read the information about the aggregation from aggregation properties
        root = AggregationPart.from_tree_dict(current_task["aggregation"],
                                              properties.template_parts,
                                              Transformation.from_matrix(current_task["root_transformation"]))
        parts_by_id = {
            p.tags["id"]: p for p in root.traverse()
        }

        placed_parts_aggregation = [parts_by_id[i] for i in current_task["pre_placed_parts"]]
        future_parts_aggregation = [parts_by_id[i] for i in current_task["construction_order"]]

        scene = SimulatedAssemblyScene(properties.template_parts, use_part_bounding_boxes=True)
        scene.initialize_scene(env)

        spawn_position = scene.pickup_table_surface_center.transform(Transformation()).translation
        target_position = scene.place_table_surface_center.transform(Transformation()).translation
        transformation_structure_to_world: Transformation = Transformation.from_pos_euler(
            target_position, np.array([0, 0, math.pi / 2]))

        # The parts already placed in the scene
        placed_parts = [Part(part.base_part, transformation_structure_to_world.transform(part.pose))
                        for part in placed_parts_aggregation]
        # The parts that need to be placed
        future_parts = [Part(part.base_part, transformation_structure_to_world.transform(part.pose),
                             Transformation(spawn_position + np.array([0, 0, part.base_part.mesh.extents[2] / 2]),
                                            transformation_structure_to_world.transform(part.pose).rotation))
                        for part in future_parts_aggregation]

        for part in placed_parts:
            scene.spawn_part(part)

        robot = scene.robot
        robot.arm.set_joint_mode(JointMode.POSITION_CONTROL)
        robot.gripper.set_joint_mode(JointMode.POSITION_CONTROL)

        actuate_gripper(env, robot.gripper, 0.7)
        for current_part in future_parts:
            scene.spawn_part(current_part)
            scene.update_target_marker(current_part)

            # Check whether the part is flipped (a different grasping position is needed if the part is flipped)
            if current_part.pose.matrix[2, 2] > 0:
                gripper_pose_part_frame = Transformation.from_pos_euler(
                    position=np.array([0.0675, 0.0, 0.16]), euler_angles=np.array([np.pi / 2, -np.pi / 2, -np.pi / 2]),
                    sequence="XYZ")
                prepick_pose_part_frame = Transformation.from_pos_euler(position=np.array([0, 0, 0.2]))
                preplace_pose_part_frame = Transformation.from_pos_euler(position=np.array([0, 0, 0.2]))
            else:
                gripper_pose_part_frame = Transformation.from_pos_euler(
                    position=np.array([0.029, 0.0, -0.16]), euler_angles=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2]),
                    sequence="XYZ")
                prepick_pose_part_frame = Transformation.from_pos_euler(position=np.array([0, 0, -0.2]))
                preplace_pose_part_frame = Transformation.from_pos_euler(position=np.array([0, 0, -0.2]))

            pick_pose = current_part.pose.transform(gripper_pose_part_frame)
            prepick_pose = current_part.pose.transform(prepick_pose_part_frame.transform(gripper_pose_part_frame))
            place_pose = current_part.target_pose.transform(gripper_pose_part_frame)
            preplace_pose = current_part.target_pose.transform(preplace_pose_part_frame.transform(gripper_pose_part_frame))
            move_arm(env, robot, prepick_pose, 40)
            move_arm(env, robot, pick_pose, 25)
            actuate_gripper(env, robot.gripper, 0.8)
            move_arm(env, robot, prepick_pose, 25)
            move_arm(env, robot, preplace_pose, 40)
            # Intermediate step to ensure that the gripper move down in a somewhat straight line
            move_arm(env, robot, Transformation(0.5 * (preplace_pose.translation + place_pose.translation), place_pose.rotation))
            move_arm(env, robot, place_pose, 25)
            # if not args.intelligent or target_part.connection_point_to_parent is None:
            #     pr_env.move_arm_path([prepick_pose, preplace_pose, place_pose], visualize_path=False,
            #                          execution_step_sleep=EXECUTION_STEP_SLEEP,
            #                          callbacks=execution_callbacks + [check_collision_callback])
            # else:
            #     parent_part = target_part.connection_point_to_parent.connected_to.parent_part
            #     target_pose_parent_frame = parent_part.pose.transform(
            #         target_part.pose, inverse=True)
            #     pose = placed_parts[parent_part].get_pose()
            #     parent_pose = Transformation(pose[:3], pose[3:])
            #     target_pose_i = parent_pose.transform(target_pose_parent_frame)
            #     place_pose_i = target_pose_i.transform(gripper_pose_part_frame)
            #     preplace_pose_i = target_pose_i.transform(preplace_pose_part_frame.transform(gripper_pose_part_frame))
            #     pr_env.move_arm_path([prepick_pose, preplace_pose_i, place_pose_i], visualize_path=False,
            #                          execution_step_sleep=EXECUTION_STEP_SLEEP,
            #                          callbacks=execution_callbacks + [check_collision_callback])
            #     pr_env.move_arm_path([place_pose], visualize_path=False,
            #                          execution_step_sleep=EXECUTION_STEP_SLEEP,
            #                          callbacks=execution_callbacks)
            actuate_gripper(env, robot.gripper, 0.7)
            move_arm(env, robot, preplace_pose, 25)

            print(f"Reached pos: {current_part.pose.translation}")
            print("")

        sleep(1000)
    finally:
        env.shutdown()

