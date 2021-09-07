import trimesh.viewer
import numpy as np

from pathlib import Path

import pybullet as p
import time
import pybullet_data

from aggregation import random_aggregation, AggregationProperties
from generate_tasks import compute_lowest_point

if __name__ == "__main__":
    properties_path = Path.cwd().parent / "example" / "rhino_export" / "box"
    properties = AggregationProperties.load(properties_path)

    template_parts = [p for p in properties.template_parts if p.name == "short"]

    scene_root = random_aggregation(template_parts, max_part_count=200,
                                    global_target_shape=properties.global_target_shape,
                                    cache_dir=properties_path / "probe_cache")
    scene_parts = scene_root.traverse()
    min_position_z = compute_lowest_point(scene_parts)

    scene = trimesh.Scene()
    for part in scene_parts:
        scene.add_geometry(part.base_part.mesh.copy().apply_transform(part.pose.matrix))

    scene.add_geometry(properties.global_target_shape)

    trimesh.viewer.SceneViewer(scene)

    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    print(p.getAPIVersion())

    mesh_coll = {}

    for m in ["short", "long"]:
        mesh_filenames = sorted([str(f) for f in (properties_path / m).iterdir()])

        coll = p.createCollisionShapeArray([p.GEOM_MESH] * len(mesh_filenames), fileNames=mesh_filenames,
                                           meshScales=np.ones((len(mesh_filenames), 3)) * 0.001)
        mesh_coll[m] = coll

    for part in scene_parts:
        pos = part.pose.translation
        test_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=mesh_coll[part.base_part.name],
                                      basePosition=pos + np.array([0, 0, - min_position_z]),
                                      baseOrientation=part.pose.quaternion)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")

    print("Start")
    input("Press enter to continue...")
    print("Go")
    for i in range(10000):
        if i in range(0, 5):
            print("Step")
        p.stepSimulation()
        time.sleep(1. / 240.)
    p.disconnect()
