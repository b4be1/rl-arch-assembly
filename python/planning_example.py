import logging
from pathlib import Path
from time import sleep

import numpy as np
import trimesh
from pybullet_planning import Attachment, create_attachment, get_sample_fn, get_distance_fn, get_extend_fn, \
    get_collision_fn, get_joint_positions, check_initial_end, MAX_DISTANCE, plan_joint_motion, plan_lazy_prm, BASE_LINK
from pybullet_planning.motion_planners.rrt_star import rrt_star
from scipy.spatial.transform import Rotation

import pybullet
import pybullet_planning

from assembly_gym.environment.generic import JointMode
from assembly_gym.environment.pybullet import PybulletEnvironment
from assembly_gym.environment.simulation import ShapeTypes
from assembly_gym.util import Transformation


def add_box():
    box_extents = np.array([0.2, 0.25, 0.05])
    obstacle_collision_shape = env.create_collision_shape(ShapeTypes.BOX, box_extents=[box_extents])
    obstacle_visual_shape = env.create_visual_shape(ShapeTypes.BOX, box_extents=[box_extents],
                                                    rgba_colors=(1.0, 0.0, 0.0, 1.0))
    obstacle = env.add_simple_object(obstacle_visual_shape, obstacle_collision_shape)
    return obstacle


def add_part():
    mesh_path = Path(__file__).parent.parent / "example" / "meshes" / "sl" / "decomposed"
    part_meshes = []
    for shape_file_name in mesh_path.iterdir():
        with shape_file_name.open() as shape_file:
            part_meshes.append(trimesh.load(shape_file, file_type="obj").apply_scale(1e-3))
    part_collision_shape = env.create_collision_shape(ShapeTypes.MESH, mesh_data=part_meshes, mesh_scale=1e-3,
                                                      use_mesh_bounding_boxes=True)
    part = env.add_simple_object(collision_shape=part_collision_shape, mass=1.0)
    return part


def grasp_part(robot, part, environment) -> bool:
    """
    Move the gripper to the part and grasp it.

    :return:    whether grasping the part was successful
    """
    gripper = robot.gripper

    # Make the part static to ensure that the part remains at the spawn position
    part.set_static(True)
    part.set_collidable(False)
    # Make the robot static to allow setting its joint angles
    robot.arm.set_static(True)
    robot.gripper.set_static(True)
    successful = position_gripper_at_part(robot, part) <= 1e-2

    # Set the gripper position to almost touch the part
    # --> we cannot just close the gripper since the fingers would just get stuck in the part
    gripper.move_to_joint_positions(np.array([0.65]))
    # Enable dynamics again to grasp the part
    part.set_collidable(True)
    robot.gripper.set_static(False)

    gripper_joint_mode = gripper.get_joint_mode()
    gripper.set_joint_mode(JointMode.POSITION_CONTROL)
    gripper.set_joint_target_positions(np.array([0.9]))
    for i in range(5):
        environment.step()
    gripper.set_joint_mode(gripper_joint_mode)
    robot.arm.set_static(False)
    part.set_static(False)
    return successful


def position_gripper_at_part(robot, part) -> float:
    """
    Place the gripper at the part so that the fingers are around on of the middle "spikes" of the part. The position
    of the gripper will be offsetted uniformly randomly (inside the limits specified by self.__grasp_offset_limits)
    parallel to the "spikes" and along the height of the parts.

    :return:        the error in the gripper position
    """
    grasp_offset = np.array([0.01, -0.135, -0.045])
    gripper_orientation = np.array([0, 0, np.pi / 2])
    gripper_pose_part_frame = Transformation(translation=grasp_offset,
                                             rotation=Rotation.from_euler("XYZ", angles=gripper_orientation))
    gripper_pose_world_frame = part.pose.transform(gripper_pose_part_frame)

    gripper = robot.gripper
    gripper_pos = np.array([0.4])
    gripper.move_to_joint_positions(gripper_pos)
    gripper.set_joint_target_positions(gripper_pos)

    gripper_position_error = robot.place_gripper_at(gripper_pose_world_frame)
    return gripper_position_error


def plan_joint_motion_rrt_star(body, joints, end_conf, obstacles=[], attachments=[],
                               self_collisions=True, disabled_collisions=set(), extra_disabled_collisions=set(),
                               weights=None, resolutions=None, max_distance=MAX_DISTANCE, custom_limits={},
                               diagnosis=False, **kwargs):
    """call rrt_star to plan a joint trajectory from the robot's **current** conf to ``end_conf``.
    """
    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(body, joints, weights=weights)
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles=obstacles, attachments=attachments,
                                    self_collisions=self_collisions,
                                    disabled_collisions=disabled_collisions,
                                    extra_disabled_collisions=extra_disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance)

    start_conf = get_joint_positions(body, joints)

    if not check_initial_end(start_conf, end_conf, collision_fn, diagnosis=diagnosis):
        return None
    radius = 0.1
    return rrt_star(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, radius, max_iterations=1000, informed=False, **kwargs)

def plan_joint_motion_lazy_prm(body, joints, end_conf, obstacles=[], attachments=[],
                               self_collisions=True, disabled_collisions=set(), extra_disabled_collisions=set(),
                               weights=None, resolutions=None, max_distance=MAX_DISTANCE, custom_limits={},
                               diagnosis=False, **kwargs):
    """call rrt_star to plan a joint trajectory from the robot's **current** conf to ``end_conf``.
    """
    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(body, joints, weights=weights)
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles=obstacles, attachments=attachments,
                                    self_collisions=self_collisions,
                                    disabled_collisions=disabled_collisions,
                                    extra_disabled_collisions=extra_disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance)

    start_conf = get_joint_positions(body, joints)

    if not check_initial_end(start_conf, end_conf, collision_fn, diagnosis=diagnosis):
        return None
    return plan_lazy_prm(start_conf, end_conf, sample_fn, extend_fn, collision_fn)


env = PybulletEnvironment(headless=False, real_time_factor=0.1)
env.initialize(0.05)
pybullet.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=160, cameraPitch=-40,
                                    cameraTargetPosition=[0, 0.4, 0.0])
try:
    robot = env.add_ur10_robot("ur10")
    part = add_part()
    obstacle_pose = Transformation(np.array([0.0, 0.8, 0.5]))
    obstacle = add_box()
    obstacle.set_pose(obstacle_pose)

    robot.gripper.set_joint_mode(JointMode.POSITION_CONTROL)
    robot.arm.set_joint_mode(JointMode.POSITION_CONTROL)
    robot_pb = robot.arm.wrapped_body.unique_id
    joints = [j.wrapped_joint.joint_index for j in robot.arm.joints]

    while True:
        part_pose = Transformation.from_pos_euler(np.array([0.4, 0.8, 0.5]), np.array([-np.pi / 2, 0, 0]))
        postion_noise = np.random.uniform(-0.02, 0.02, 3)
        part.set_pose(Transformation(part_pose.translation + postion_noise, part_pose.rotation))
        grasp_part(robot, part, env)
        initial_pose = robot.arm.wrapped_body.links["wrist_3_link"].pose
        print("Initial position: {}".format(robot.arm.wrapped_body.links["wrist_3_link"].pose.translation))
        target_pose = Transformation(initial_pose.translation * np.array([-1, 1, 1]), robot.gripper.pose.rotation)
        target_joint_pos = robot.solve_ik(target_pose)[:6]
        part_attachment = create_attachment(robot.arm.wrapped_body.unique_id,
                                            robot.arm.wrapped_body.links["rh_p12_rn_r2"].link_index,
                                            part.wrapped_body.unique_id)
        reset_checkpoint = env.physics_client.call(pybullet.saveState)  # TODO: This should be integrated into environment
        extra_disabled_collisions = []
        for gripper_link in robot.gripper.wrapped_body.links.values():
            extra_disabled_collisions.append(((robot.gripper.wrapped_body.unique_id, gripper_link.link_index),
                                              (part.wrapped_body.unique_id, BASE_LINK)))
        solution = plan_joint_motion_rrt_star(robot_pb, joints, target_joint_pos, diagnosis=True,
                                              extra_disabled_collisions=extra_disabled_collisions,
                                              # obstacles=[env.ground_plane.unique_id])#,
                                              obstacles=[obstacle.wrapped_body.unique_id, env.ground_plane.unique_id],
                                              attachments=[part_attachment])
        env.physics_client.call(pybullet.restoreState, stateId=reset_checkpoint)
        # part.set_pose(part_pose)
        # grasp_part(robot, part, env)
        # robot.arm.set_joint_mode(JointMode.POSITION_CONTROL)

        print("Initial joint position: {}".format(robot.arm.joint_positions))
        if solution is not None:
            print(solution)
            for configuration in solution:
                robot.arm.set_joint_target_positions(configuration)
                env.step()
            print("Joint positions: {}".format(robot.arm.joint_positions))
            print("Target joint positions: {}".format(target_joint_pos))
        else:
            logging.warning("Planner did not find a solution")
        print("Final position: {}".format(robot.arm.wrapped_body.links["wrist_3_link"].pose.translation))
        for _ in range(10):
            env.step()
finally:
    env.shutdown()
