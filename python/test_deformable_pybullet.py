import time
import pybullet_data

import numpy as np

import pybullet
from pybullet_wrapper import PhysicsClient
from pybullet_wrapper.multibody import URDFBody
from pybullet_wrapper.simulation_object import JointControlMode

if __name__ == "__main__":
    pc = PhysicsClient()
    pc.connect_gui()

    pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)

    ts = 0.005

    pc.set_additional_search_path(pybullet_data.getDataPath())
    pc.gravity = np.array([0, 0, -9.81])
    pc.time_step = ts
    plane = URDFBody("plane.urdf", base_position=np.array([0, 0, 0]))

    sensor_body = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.015, 0.015, 0.001])

    gripper = URDFBody("../pybullet/models/rh_p12_rn.urdf", base_position=np.array([0, 0, 0.1]),
                       flags=pybullet.URDF_USE_SELF_COLLISION, use_fixed_base=True)

    sensor_pos_l2 = np.array([0, 0.012, 0.015])
    sensor_quat_l2 = pybullet.getQuaternionFromEuler([-np.pi / 2, 0, 0])

    sensor_pos_r2 = np.array([0, -0.012, 0.015])
    sensor_quat_r2 = pybullet.getQuaternionFromEuler([np.pi / 2, 0, 0])

    l2_link = gripper.links["rh_p12_rn_l2"]
    r2_link = gripper.links["rh_p12_rn_r2"]

    l2_pos, l2_quat, *_ = pybullet.getLinkState(gripper.unique_id, l2_link.link_index)
    r2_pos, r2_quat, *_ = pybullet.getLinkState(gripper.unique_id, r2_link.link_index)

    sensor_l2_world_pos, sensor_l2_world_quat = pybullet.multiplyTransforms(l2_pos, l2_quat, sensor_pos_l2, sensor_quat_l2)
    sensor_r2_world_pos, sensor_r2_world_quat = pybullet.multiplyTransforms(r2_pos, r2_quat, sensor_pos_r2, sensor_quat_r2)

    elastomer_l2_world_pos, elastomer_l2_world_quat = pybullet.multiplyTransforms(sensor_l2_world_pos, sensor_l2_world_quat,
                                                                                  [0, 0, 0.02], [0, 0, 0, 1])
    elastomer_r2_world_pos, elastomer_r2_world_quat = pybullet.multiplyTransforms(sensor_r2_world_pos, sensor_r2_world_quat,
                                                                                  [0, 0, 0.02], [0, 0, 0, 1])

    elastomer_l2 = pybullet.loadSoftBody(fileName="../pybullet/models/elastomer.obj", basePosition=elastomer_l2_world_pos,
                                         baseOrientation=elastomer_l2_world_quat, mass=0.005,
                                         useNeoHookean=1, useBendingSprings=1, useMassSpring=1, springElasticStiffness=40,
                                         springDampingStiffness=.1, springDampingAllDirections=1, useSelfCollision=0,
                                         frictionCoeff=1.0, useFaceContact=1, collisionMargin=0.001)
    elastomer_r2 = pybullet.loadSoftBody(fileName="../pybullet/models/elastomer.obj", basePosition=elastomer_r2_world_pos,
                                         baseOrientation=elastomer_r2_world_quat, mass=0.005,
                                         useNeoHookean=1, useBendingSprings=1, useMassSpring=1, springElasticStiffness=40,
                                         springDampingStiffness=.1, springDampingAllDirections=1, useSelfCollision=0,
                                         frictionCoeff=1.0, useFaceContact=1, collisionMargin=0.001)

    elastomer_l2_pos, elastomer_l2_quat = pybullet.multiplyTransforms(sensor_pos_l2, sensor_quat_l2, [0, 0, 0.02], [0, 0, 0, 1])
    elastomer_r2_pos, elastomer_r2_quat = pybullet.multiplyTransforms(sensor_pos_r2, sensor_quat_r2, [0, 0, 0.02], [0, 0, 0, 1])
    anchor_points = {
        1: [-0.015, 0.015, 0],
        3: [0.015, 0.015, 0],
        5: [-0.015, -0.015, 0],
        7: [0.015, -0.015, 0]
    }
    for i, p in anchor_points.items():
        pybullet.createSoftBodyAnchor(elastomer_l2, i, gripper.unique_id, l2_link.link_index, p)
        pybullet.createSoftBodyAnchor(elastomer_r2, i, gripper.unique_id, r2_link.link_index, p)

    sphere_coll = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=0.01)
    sphere = pybullet.createMultiBody(basePosition=np.array([0, 0, 0.24]), baseCollisionShapeIndex=sphere_coll)

    pybullet.changeDynamics(sphere, -1, lateralFriction=1.0, spinningFriction=1.0, rollingFriction=1.0)

    for j in gripper.revolute_joints:
        j.control_mode = JointControlMode.POSITION_CONTROL

    r1 = gripper.joints["rh_r1"]
    r2 = gripper.joints["rh_r2"]
    l1 = gripper.joints["rh_l1"]
    l2 = gripper.joints["rh_l2"]

    r1.target_position = 0.45
    r2.target_position = 1.0
    l1.target_position = 0.45
    l2.target_position = 1.0

    start = time.time()
    try:
        next_step = time.time()
        i = 0
        while True:
            if i == 100:
                pybullet.changeDynamics(sphere, -1, mass=0.005)
            elif i == 1500:
                r1.target_position = 0.0
                l1.target_position = 0.0
            now = time.time()
            time.sleep(max(0.0, next_step - now))
            next_step += ts
            pc.step_simulation()
            i += 1
    finally:
        pc.disconnect()
