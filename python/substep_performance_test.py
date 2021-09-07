import random
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import pybullet
import pybullet_data

import numpy as np

from aggregation import AggregationProperties
from pybullet_wrapper import PhysicsClient
from pybullet_wrapper.simulation_body import URDFBody
from pybullet_wrapper.simulation_object import JointControlMode

if __name__ == "__main__":
    times = []
    steps = 100
    ts = 0.05
    num_bodies = 5
    substeps = 10
    virtual_substep_mode = True
    if virtual_substep_mode:
        inner_substeps = 1
        total_time_step = ts / substeps
    else:
        inner_substeps = substeps
        total_time_step = ts
    body_interval = steps // num_bodies // 2 if num_bodies > 0 else None
    bounding_box_optimization = True

    properties_path = Path.cwd().parent / "example" / "rhino_export" / "box"
    properties = AggregationProperties.load(properties_path)

    pc = PhysicsClient()
    pc.connect_direct()
    pc.call(pybullet.setPhysicsEngineParameter, numSubSteps=inner_substeps)

    pc.set_additional_search_path(pybullet_data.getDataPath())
    pc.gravity = np.array([0, 0, -9.81])
    pc.time_step = total_time_step
    pc.add_body(URDFBody("plane.urdf", base_position=np.array([0, 0, 0])))

    robot = URDFBody("../pybullet/models/ur10.urdf", base_position=np.array([0, 0, 0]))

    pc.add_body(robot)

    mesh_coll = {}

    for m in ["short", "long"]:
        if not bounding_box_optimization:
            mesh_filenames = sorted([str(f) for f in (properties_path / m).iterdir()])
            coll = pybullet.createCollisionShapeArray([pybullet.GEOM_MESH] * len(mesh_filenames),
                                                      fileNames=mesh_filenames,
                                                      meshScales=np.ones((len(mesh_filenames), 3)) * 0.001)
            mesh_coll[m] = coll
        else:
            meshes = [p for p in properties.template_parts if p.name == m][0].sub_meshes
            dims = []
            pos = []
            for sub_mesh in meshes:
                min_pos = np.min(sub_mesh.vertices, axis=0)
                max_pos = np.max(sub_mesh.vertices, axis=0)
                dims.append((max_pos - min_pos) / 2)
                pos.append((max_pos + min_pos) / 2)
            coll = pybullet.createCollisionShapeArray([pybullet.GEOM_BOX] * len(meshes), halfExtents=dims,
                                                      collisionFramePositions=pos)
            mesh_coll[m] = coll

    colls = list(mesh_coll.values())

    for j in robot.revolute_joints:
        j.control_mode = JointControlMode.VELOCITY_CONTROL
        j.target_velocity = 0

    tmp = NamedTemporaryFile(mode="wb")

    s = time.time()
    pybullet.saveBullet(bulletFileName=tmp.name)
    print("Save: {}".format(time.time() - s))

    step_times = []

    for t in range(50):
        s = time.time()
        pc.call(pybullet.restoreState, fileName=tmp.name)
        print("Restore: {}".format(time.time() - s))
        start = time.time()
        bodies = []
        for i in range(steps):
            if body_interval is not None and i % body_interval == 0 and len(bodies) < num_bodies:
                coll = random.choice(colls)
                bodies.append(pybullet.createMultiBody(baseMass=1, baseCollisionShapeIndex=coll,
                                                       basePosition=np.array([0, 0, 2])))
            for j in robot.revolute_joints:
                j.target_velocity = max(min(j.target_velocity + np.random.uniform(-1, 1), 20), -20)

            step_start = time.time()
            for j in range(substeps // inner_substeps):
                pc.step_simulation()
            step_times.append(time.time() - step_start)
        delta = time.time() - start
        for b in bodies:
            pc.call(pybullet.removeBody, b)
        times.append(delta)
        print("{:0.5f} (speedup: {:0.5f} real time)".format(delta, steps * ts / delta))
    pc.disconnect()
    print("Average: {:0.5f} (speedup: {:0.5f} real time, {:0.8f} step time)".format(
        np.average(times), steps * ts / np.average(times), np.mean(step_times)))
