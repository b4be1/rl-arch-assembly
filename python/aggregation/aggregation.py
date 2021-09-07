"""
This file provides functionality to compute Grasshopper Wasp like aggregations
"""
import hashlib
import logging
import random
from pathlib import Path
from time import time
from typing import Sequence, List, Tuple, Optional
import numpy as np

import trimesh.viewer
import trimesh.creation
import trimesh.boolean
import trimesh.sample
import trimesh.intersections
import trimesh.visual
from trimesh import Trimesh
from trimesh.repair import fill_holes

from trimesh.collision import CollisionManager

from assembly_gym.util import Transformation
from .template_connection_point import AllowedConnection
from .util import calculate_aggregation_transform
from .aggregation_part import AggregationPart
from .template_part import TemplatePart

logger = logging.getLogger(__file__)


def intersect(meshes: List[Trimesh]) -> Trimesh:
    """
    Computes the boolean intersection of a list of meshes in a binary tree fashion
    :param meshes:  Meshes to compute intersection of
    :return: Mesh whose interior points are a subset of all given meshes
    """
    assert len(meshes) > 0
    if len(meshes) == 2:
        # Note: the Blender engine appears to be broken
        m = trimesh.boolean.intersection(meshes, engine="scad")
        trimesh.repair.fill_holes(m)
        return m
    elif len(meshes) == 1:
        return meshes[0]
    else:
        s = len(meshes) // 2
        m1 = intersect(meshes[:s])
        m2 = intersect(meshes[s:])
        return intersect([m1, m2])


def compute_connection_signature(allowed_connections: Tuple[AllowedConnection]) -> str:
    """
    Computes a string signature for the given tuple of allowed connections
    :param allowed_connections:
    :return:
    """
    return hashlib.sha256(str(allowed_connections).encode("utf-8")).hexdigest()


def random_aggregation(parts: Sequence[TemplatePart], global_target_shape: trimesh.Trimesh, max_part_count: int,
                       seed: Optional[int] = None, cache_dir: Optional[Path] = None) -> AggregationPart:
    """
    Computes a random aggregation of the given parts
    :param parts:               Template parts to build aggregation of.
    :param global_target_shape: Mesh that describes the target shape of the structure. The aggregation will make sure
                                that all parts fully lie within this mesh.
    :param max_part_count:      Maximum number of parts to place.
    :param seed:                Random seed for this aggregation.
    :param cache_dir:           Directory in which the probe part cache shall be stored.
    :return: Root node of the aggregation
    """

    logger.debug("Generating aggregation...")
    start_time = time()

    if seed is not None:
        random.seed(seed)
        np.random.seed(random.randint(0, 2 ** 32 - 1))

    logger.debug("Determining probe parts...")
    # Determine a probe part for each connection point class. The probe part of a connection point is defined as the
    # intersection of all possible parts in all poses that can be attached to that connection point. Having such a probe
    # part allows to rule out connection points in the aggregation, because when the probe part is already colliding
    # with other parts, no part will ever be able to fit at that connection point.

    parts_by_name = {
        p.name: p for p in parts
    }

    connections_per_connection_point = {
        p: {
            c: tuple(sorted(c.allowed_connections))
            for c in p.connection_points
        }
        for p in parts
    }

    connections_set = set(
        tuple(ac for ac in v if ac.part_name in parts_by_name)
        for p, cs in connections_per_connection_point.items() for v in cs.values())

    connection_probes_per_signature = {}

    for connections in connections_set:
        connection_signature = compute_connection_signature(connections)
        if cache_dir is not None and (cache_dir / "{}.obj".format(connection_signature)).exists():
            # The probe part is always stored in the cache directory with the name set according to its signature
            logger.debug("Found probe for {}".format(connection_signature))
            with (cache_dir / "{}.obj".format(connection_signature)).open() as f:
                connection_probes_per_signature[connections] = trimesh.load(f, file_type="obj").apply_scale(1e-3)
        else:
            if len(connections) > 0:
                # Find all parts that could be attached to the connection point of the given class
                meshes_to_intersect = []
                for pn, ci in connections:
                    part = parts_by_name[pn]
                    connection_point = parts_by_name[pn].connection_points[ci]

                    t = connection_point.transformation.inv.matrix
                    m = part.mesh.copy()
                    m.apply_transform(t)
                    meshes_to_intersect.append(m)

                # Intersect all of them
                logger.debug("Intersecting {} elements...".format(len(meshes_to_intersect)))
                result = intersect(meshes_to_intersect)
                logger.debug("Obtained probe part with volume {:0.4e}".format(result.volume))
                connection_probes_per_signature[connections] = result
                if cache_dir is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    with (cache_dir / "{}.obj".format(connection_signature)).open("w") as f:
                        result = result.copy()
                        result.apply_scale(1e3)
                        result.export(f, file_type="obj")
            else:
                connection_probes_per_signature[connections] = None

    connection_probes_per_connection_point = {
        p: {
            c: connection_probes_per_signature[
                tuple(ac for ac in sorted(c.allowed_connections) if ac.part_name in parts_by_name)]
            for c in p.connection_points
        }
        for p in parts
    }

    # Create a collision manager to check for collisions with the global target shape during the aggregation
    collision_manager_global_shape = CollisionManager()
    collision_manager_global_shape.add_object("global_target_shape", global_target_shape)

    # Select a random part as root and place it somewhere on the bottom plane
    root_part: TemplatePart = random.choice(parts)
    min_gap_size = min(cp.gap_size for cp in root_part.connection_points)
    root_transformation = _sample_root_transform(
        collision_manager_global_shape, global_target_shape, root_part, min_gap_size)
    root = AggregationPart(root_part, parts, root_transformation)

    # Create a collision manager to check for collisions with other parts during the aggregation
    collision_manager_parts = CollisionManager()
    collision_manager_parts.add_object("0_" + root_part.name, root_part.mesh, root_transformation.matrix)

    # Keep track of unused connection points
    open_connections = root.available_connection_points

    # Statistics
    ruled_out_by_probe = 0
    total_ruled_out = 0

    logger.debug("Starting aggregation...")
    for i in range(1, max_part_count):
        done = False
        while not done and len(open_connections) > 0:
            logger.debug("\rPlaced {: 3d} parts so far ({: 5d}) connection points are potentially still available "
                         "({}/{} ruled out by probe).".format(i, len(open_connections), ruled_out_by_probe,
                                                              total_ruled_out),
                         end="")
            # Choose a random open connection
            cp_parent_index = random.randint(0, len(open_connections) - 1)
            cp_parent = open_connections[cp_parent_index]
            parent = cp_parent.parent_part

            # Probe whether any part can fit in there at all using the probe part for this connection
            probe_mesh = connection_probes_per_connection_point[parent.base_part][cp_parent.base_connection_point]

            probe_transformation = calculate_aggregation_transform(
                cp_parent.transformation, Transformation(),
                gap_size=cp_parent.base_connection_point.gap_size * 2)

            global_shape_collision = collision_manager_global_shape.min_distance_single(
                probe_mesh, transform=probe_transformation.matrix) < cp_parent.base_connection_point.gap_size
            part_collision = collision_manager_parts.in_collision_single(
                probe_mesh, transform=probe_transformation.matrix)

            if part_collision or global_shape_collision:
                cp_parent.mark_fully_blocked()
                open_connections[cp_parent_index:cp_parent_index + 1] = []
                ruled_out_by_probe += 1
                total_ruled_out += 1
                continue

            # Select child
            new_placed_part = None
            transformation_new = None

            # Iterate through possible parts and poses in random order
            possible_connections = [(p, ci) for p, c in cp_parent.possible_connections.items() for ci in c]
            random.shuffle(possible_connections)
            for root_part, cp_child in possible_connections:
                new_placed_part = parent.add_child(cp_parent, root_part, cp_child)

                # Calculate the transformation of the new part and apply it
                transformation_new = new_placed_part.pose

                # Check that the inserted object is not in collision with any object in the scene and not in collision
                # with the global shape
                global_shape_collision = collision_manager_global_shape.min_distance_single(
                    root_part.mesh, transform=transformation_new.matrix) < cp_parent.base_connection_point.gap_size
                part_collision = collision_manager_parts.in_collision_single(
                    root_part.mesh, transform=transformation_new.matrix)
                if part_collision or global_shape_collision:
                    parent.remove_child(cp_parent)
                    cp_parent.mark_blocked(root_part, cp_child)
                    if cp_parent.is_fully_blocked:
                        open_connections[cp_parent_index:cp_parent_index + 1] = []
                        total_ruled_out += 1
                    new_placed_part = None
                else:
                    break
            if new_placed_part is not None:
                open_connections[cp_parent_index:cp_parent_index + 1] = []
                open_connections += new_placed_part.available_connection_points
                collision_manager_parts.add_object(f"{i}_{root_part.name}", root_part.mesh, transformation_new.matrix)
                done = True
        if not done:
            break

    timedelta = time() - start_time

    logger.debug("\rConstructing structure with {} elements took {:0.4f}s. {}/{} parts where ruled out by probe"
                 "                                        ".format(len(root.traverse()), timedelta, ruled_out_by_probe,
                                                                   total_ruled_out))
    return root


def _sample_root_transform(collision_manager_global_shape: CollisionManager, global_target_shape: trimesh.Trimesh,
                           root_part: TemplatePart, min_distance_to_border: float) -> Transformation:
    """
    Returns a random location for the given root part on the bottom plane of the global target shape.
    :param collision_manager_global_shape:  Collision manager of the global target shape
    :param global_target_shape:             Global target shape mesh
    :param root_part:                       Template part to place at random
    :return: Random transformation of the given root template on the bottom of the given global target structure
    """
    eps = 0.0001
    assert np.all(global_target_shape.extents - root_part.mesh.extents > 2 * (min_distance_to_border + eps)), \
        "Global target shape is too small to place root part {}".format(root_part.name)
    logger.debug("Placing first part randomly on the bottom plane...")
    done = False
    transformation = None
    vertices_min_z = np.min([v[2] for v in global_target_shape.vertices])
    plane = global_target_shape.section_multiplane(plane_origin=np.array([0, 0, vertices_min_z]),
                                                   plane_normal=np.array([0, 0, 1]), heights=[0])[0]
    while not done:
        pos_sample = plane.sample(1)[0]
        root_part_min_z = min([v[2] for v in root_part.mesh.vertices])
        root_position = np.array(
            [pos_sample[0], pos_sample[1], vertices_min_z - root_part_min_z + min_distance_to_border + eps])
        transformation = Transformation(root_position)
        if collision_manager_global_shape.min_distance_single(
                root_part.mesh, transform=transformation.matrix) >= min_distance_to_border:
            done = True
    logger.debug("Done placing first part")
    return transformation
