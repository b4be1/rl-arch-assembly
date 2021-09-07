import numpy as np
import trimesh.boolean
import scipy.spatial.transform
from trimesh import Trimesh

from assembly_gym.util import Transformation


def calculate_transformation(euler: np.ndarray, position: np.ndarray) -> np.ndarray:
    """
    Calculates the transformation matrix from a given position and euler angles (according to Coppelia's specification
    of the euler angles).

    :param euler:           the array of euler angles
    :param position:        the position as array
    :return:                the 4 x 4 homogeneous transformation matrix
    """
    rot = scipy.spatial.transform.Rotation.from_euler("xyz", euler).as_matrix()
    return np.concatenate((np.concatenate(
        (rot, position.reshape(3, 1)), axis=1), np.array([[0, 0, 0, 1]])), axis=0)


def calculate_euler(transformation: np.ndarray) -> np.ndarray:
    rot = scipy.spatial.transform.Rotation.from_matrix(transformation[:3, :3]).as_euler("xyz")
    return rot


def calculate_aggregation_transform(
        connection_point_1_transform: Transformation, connection_point_2_transform: Transformation,
        gap_size: float = 0.0) -> Transformation:
    """
    Calculates the transformation matrix to connect part 2 to part 1 at the given connection points

    :param connection_point_1_transform:        the transformation of the connection point of part 1
    :param connection_point_2_transform:        the transformation of the connection point of part 2
    :param gap_size:                            size of the gap between the two connection points
    :return:                                    the transformation matrix
    """
    # Rotate 180 degrees around x and apply gap size
    invert_z_transform = Transformation(np.array([0, 0, gap_size]), np.array([1, 0, 0, 0]))
    return connection_point_1_transform.transform(invert_z_transform).transform(connection_point_2_transform.inv)


def mesh_contains(mesh_outer: Trimesh, mesh_inner: Trimesh) -> bool:
    """
    Check whether one mesh is completely inside another mesh

    :param mesh_outer:      the potential outer mesh
    :param mesh_inner:      the potential inner mesh
    :return:                True iff inner mesh is completely inside outer mesh
    """
    mesh_intersection = trimesh.boolean.intersection([mesh_inner, mesh_outer], engine="scad")
    return mesh_intersection.volume == mesh_inner.volume
