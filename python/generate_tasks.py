import argparse
import json
import logging
import math
import shutil
import sys
from logging import StreamHandler
from pathlib import Path
from typing import List, Dict, Sequence, Iterable

import trimesh.viewer
import numpy as np
import pybullet as p
import pybullet_data

from aggregation import random_aggregation, AggregationPart
from aggregation.aggregation_properties import AggregationProperties
from assembly_gym.util import Transformation

logger = logging.getLogger(__file__)


def simulate(parts: Sequence[AggregationPart], mesh_coll: Dict[str, int], simulation_steps: int) \
        -> Dict[AggregationPart, Transformation]:
    parts_pybullet = {}

    for part in parts:
        parts_pybullet[part] = p.createMultiBody(
            baseMass=1, baseCollisionShapeIndex=mesh_coll[part.base_part.name], basePosition=part.position,
            baseOrientation=part.pose.quaternion)

    for _ in range(simulation_steps):
        p.stepSimulation()

    poses_pb = {k: p.getBasePositionAndOrientation(v) for k, v in parts_pybullet.items()}
    poses = {k: Transformation(v[0], v[1]) for k, v in poses_pb.items()}

    # Cleanup simulator
    for pybullet_part in parts_pybullet.values():
        p.removeBody(pybullet_part)

    return poses


def find_unstable_parts(simulation_result: Dict[AggregationPart, Transformation]) -> List[AggregationPart]:
    # Remove all parts whose position or orientation changed
    parts_to_remove = []
    for part, final_pose in simulation_result.items():
        expected_position = part.pose.translation
        expected_rotation = part.pose.rotation
        difference = expected_rotation.inv() * final_pose.rotation
        angular_error = difference.magnitude()
        position_error = np.linalg.norm(expected_position - final_pose.translation)
        if angular_error > 0.01 or position_error > 0.01:
            parts_to_remove.append(part)

    return parts_to_remove


def compute_lowest_point(parts: Iterable[AggregationPart]) -> float:
    min_position_z = np.inf
    for part in parts:
        v = part.base_part.mesh.vertices
        vertices_transformed = part.pose.transform(v)
        min_position_z = min(min_position_z, np.min(vertices_transformed[:, 2]))
    return min_position_z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates random tasks and stores them in the specified directory.")
    parser.add_argument("output_directory", type=str, help="Directory to store aggregations in.")
    parser.add_argument("aggregation_components_path", type=str,
                        help="Directory in which aggregation components are stored.")
    parser.add_argument("-n", type=int, help="Number of aggregations to create.", default=10)
    parser.add_argument("--max-parts", type=int, help="Maximum number of parts to use for each aggregation.",
                        default=50)
    parser.add_argument("--min-parts", type=int, help="Maximum number of parts to use for each aggregation.",
                        default=30)
    parser.add_argument("--min-parts-to-place", type=int, default=None,
                        help="Minimum number of parts the robot has to place (the rest will be pre-placed).")
    parser.add_argument("--max-parts-to-place", type=int, default=None,
                        help="Maximum number of parts the robot has to place (the rest will be pre-placed).")
    parser.add_argument("--view-structures", action="store_true",
                        help="Shows each of the structures before testing them for stability.")
    parser.add_argument("-p", "--parts", type=str, nargs="+", help="Names of the parts to use. Default: all.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("-s", "--skip-stability-check", action="store_true",
                        help="Do not check structures for stability.")

    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',
                        level=logging.INFO)
    logger.addHandler(StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO if not args.verbose else logging.DEBUG)

    if args.min_parts_to_place is not None and args.max_parts_to_place is not None:
        assert args.min_parts_to_place <= args.max_parts_to_place

    simulation_time = 3.0
    count = args.n
    max_part_count = args.max_parts
    min_part_count = args.min_parts

    output_dir = Path(args.output_directory)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    properties_path = Path(args.aggregation_components_path)
    properties = AggregationProperties.load(properties_path)

    # Configure pybullet
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)

    simulation_steps = int(math.ceil(simulation_time / p.getPhysicsEngineParameters()["fixedTimeStep"]))

    template_parts = properties.template_parts
    if args.parts is not None:
        template_parts = [p for p in template_parts if p.name in args.parts]
    assert len(template_parts) > 0

    mesh_coll = {}
    for tp in template_parts:
        mesh_filenames = sorted([str(f) for f in (properties_path / tp.name).iterdir()])

        coll = p.createCollisionShapeArray([p.GEOM_MESH] * len(mesh_filenames), fileNames=mesh_filenames,
                                           meshScales=np.ones((len(mesh_filenames), 3)) * 0.001)
        mesh_coll[tp.name] = coll

    for i in range(count):
        logger.info("Generating aggregation {}/{}".format(i + 1, count))
        structure_ok = False

        root = None
        parts = None
        parts_to_place_count = None

        while not structure_ok:
            # Compute a random aggregation
            root = random_aggregation(template_parts, max_part_count=max_part_count,
                                      global_target_shape=properties.global_target_shape,
                                      cache_dir=properties_path / "probe_cache")

            parts = root.traverse()

            # Make sure all parts are on the ground
            min_position_z = compute_lowest_point(parts)

            for part in parts:
                part.pose.translation[2] -= min_position_z

            if args.view_structures:
                scene = trimesh.Scene()
                for part in parts:
                    scene.add_geometry(part.base_part.mesh.copy().apply_transform(part.pose.matrix))

                global_target_shape_transform = np.eye(4)
                global_target_shape_transform[2, 3] = -min_position_z
                scene.add_geometry(properties.global_target_shape, transform=global_target_shape_transform)
                trimesh.viewer.SceneViewer(scene)

            # Check stability
            logger.debug("Obtained aggregation, checking stability...")
            stable = args.skip_stability_check
            abort = False
            final_poses = None
            while not stable and not abort:
                parts_pybullet = {}

                parts = root.traverse()

                final_poses = simulate(parts, mesh_coll, simulation_steps)
                parts_to_remove = find_unstable_parts(final_poses)

                # Iterate until no parts move anymore
                stable = len(parts_to_remove) == 0
                abort = len(parts) < min_part_count

                if root not in parts_to_remove:
                    for part in parts_to_remove:
                        parent_connection_point = part.connection_point_to_parent.connected_to
                        parent = parent_connection_point.parent_part
                        parent.remove_child(parent_connection_point)
                    logger.debug("Removed {} unstable part(s).".format(len(parts_to_remove)))
                else:
                    logger.debug("Root has moved, scraping structure.")
                    abort = True

            parts = root.traverse()

            structure_ok = len(parts) >= min_part_count

            max_parts_to_place = len(parts)
            if args.max_parts_to_place is not None:
                max_parts_to_place = min(max_parts_to_place, args.max_parts_to_place)

            min_parts_to_place = len(parts)
            if args.min_parts_to_place is not None:
                min_parts_to_place = args.min_parts_to_place
                structure_ok = structure_ok and min_parts_to_place <= len(parts)

            if structure_ok:
                logger.debug("Obtained stable structure with {} part(s).".format(len(parts)))

                logger.debug("Checking constructability...")
                parts = sorted(parts, key=lambda part: part.position[2])
                for j, part in enumerate(parts):
                    part.tags["id"] = j
                    part.tags["settled_pose"] = {
                        "pos": tuple(final_poses[part].translation),
                        "quat": tuple(final_poses[part].quaternion)
                    }

                parts_to_place_count = np.random.randint(min_parts_to_place, max_parts_to_place + 1)

                logger.debug("Attempting to place {} parts".format(parts_to_place_count))

                for j in range(len(parts) - parts_to_place_count, len(parts)):
                    part_no = j - (len(parts) - parts_to_place_count) + 1
                    logger.debug("Placing part {}/{}".format(part_no, parts_to_place_count))
                    final_poses_constr = simulate(parts, mesh_coll, simulation_steps)
                    unstable_parts = find_unstable_parts(final_poses_constr)
                    structure_ok = structure_ok and len(unstable_parts) == 0
                    if not structure_ok:
                        logger.debug(
                            "Constructing structure failed in step {}/{}".format(part_no, parts_to_place_count))
                        break

            else:
                logger.debug("Not enough parts are left ({}). Retrying...".format(len(parts)))

        gts_center = properties.global_target_shape.centroid
        root_pos = root.pose.translation.copy()
        root_pos[:2] -= gts_center[:2]
        root_pos[2] += np.mean([cp.gap_size for tp in template_parts for cp in tp.connection_points])
        root_pose_gts_bottom_frame = Transformation(root_pos, root.pose.quaternion)

        task_dict = {
            "aggregation": root.tree_dict(),
            "pre_placed_parts": [part.tags["id"] for part in parts[:-parts_to_place_count]],
            "construction_order": [part.tags["id"] for part in parts[-parts_to_place_count:]],
            "root_transformation": root_pose_gts_bottom_frame.matrix.tolist()
        }
        with (output_dir / "{:04d}.json".format(i)).open("w") as f:
            json.dump(task_dict, f)

    p.disconnect()
