import logging
from pathlib import Path
from typing import Union, cast, Tuple

import numpy as np
import pyrep
from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.objects import VisionSensor

try:
    import cupy as ncp
    import cupyx.scipy.ndimage as ndimage

    cupy_used = True
except ImportError:
    import numpy as ncp
    import scipy.ndimage as ndimage

    cupy_used = False
    logging.warn("CuPy is not available. Defaulting to NumPy.")


class DigitSensor(pyrep.objects.Object):
    """
    PyRep implementation of the Digit tactile sensor
    """

    def __init__(self, name_or_handle: Union[str, int]):
        """

        :param name_or_handle: Name or handle of the sensor in Coppeliasim
        """
        super().__init__(name_or_handle)
        # The vision sensor is always called "vision_sensor", "vision_sensor0", etc.
        vision_sensor = [obj for obj in self.get_objects_in_tree() if "vision_sensor" in obj.get_name()][0]
        # Explicitly cast to VisionSensor type to avoid typing errors
        self._vision_sensor = cast(pyrep.objects.VisionSensor, vision_sensor)

        self._resolution = self._vision_sensor.get_resolution()
        background_color = ncp.array([0.4, 0.4, 0.4])
        self._background = ncp.tile(background_color[ncp.newaxis, ncp.newaxis, :],
                                    (self._resolution[0], self._resolution[1], 1))
        self._led_directions = ncp.array([[-0.5, -0.5, -0.15],
                                          [0.5, -0.5, -0.15],
                                          [0, 1, -0.15]])
        self._led_directions /= ncp.linalg.norm(self._led_directions, axis=-1)[:, ncp.newaxis]
        self._diffuse_components = ncp.array([[0.7, 0.0, 0.0],
                                              [0.0, 0.7, 0.0],
                                              [0.0, 0.0, 0.7]])
        self._specular_components = ncp.array([[0.7, 0.0, 0.0],
                                               [0.0, 0.7, 0.0],
                                               [0.0, 0.0, 0.7]])
        self._alpha = 5
        self._kd = 0.5
        self._ks = 0.15

        self._derivative_scale = 50
        self._darkening_mask_strength = 0.1

    @staticmethod
    def create(pr: PyRep, model_path: Path, name: str = None) -> "DigitSensor":
        digit = pr.import_model(str(model_path))
        if name is not None:
            digit.set_name(name)
        return DigitSensor(digit.get_handle())

    @staticmethod
    def _phong_shading_with_background(background: ncp.ndarray, depth_map: ncp.ndarray, led_directions: ncp.ndarray,
                                       specular_components: ncp.ndarray, diffuse_components: ncp.ndarray, alpha: float,
                                       kd: float, ks: float) -> ncp.ndarray:
        """
        Compue the Phong shading of the given depth map
        :param background:          Image to put into the background
        :param depth_map:           Depth map to compute shading for
        :param led_directions:      Directions of the incoming LED light (Nx3 array)
        :param specular_components: Specular components of each LED (Nx3 array)
        :param diffuse_components:  Diffuse components of each LED (Nx3 array)
        :param alpha:               Alpha parameter
        :param kd:                  kd parameter
        :param ks:                  ks parameter
        :return: Phong shaded image
        """
        derivative_kernel_x = ncp.array([[1, 0, -1]])
        derivative_kernel_y = derivative_kernel_x.T

        depth_map_x_derivative = ndimage.convolve(depth_map, derivative_kernel_x, mode="nearest")
        depth_map_y_derivative = ndimage.convolve(depth_map, derivative_kernel_y, mode="nearest")

        V = ncp.array([0, 0, -1])
        height, width, _ = background.shape
        N = ncp.stack([depth_map_x_derivative,
                       depth_map_y_derivative,
                       -ncp.ones_like(depth_map_x_derivative)], axis=-1)
        N /= ncp.linalg.norm(N, axis=-1)[:, :, ncp.newaxis]

        corr = ncp.maximum(0.0, N.dot(led_directions.T))
        R = 2 * corr[:, :, :, ncp.newaxis] * N[:, :, ncp.newaxis, :] - led_directions[ncp.newaxis, ncp.newaxis, :, :]
        R /= ncp.linalg.norm(R, axis=-1)[:, :, :, ncp.newaxis]

        result = kd * corr[:, :, :, ncp.newaxis] * diffuse_components[ncp.newaxis, ncp.newaxis, :, :] \
                 + ks * ncp.maximum(0.0, R.dot(V))[:, :, :, ncp.newaxis] ** alpha \
                 * specular_components[ncp.newaxis, ncp.newaxis, :, :]
        result = ncp.sum(result, axis=2)

        result += background

        return result

    def get_measurements(self) -> np.ndarray:
        """
        Obtain the measurements of the tactile sensor
        :return:
        """
        depth_map = ncp.array(self._vision_sensor.capture_depth(in_meters=False)).T
        depth_map_amplified = depth_map * self._derivative_scale
        result = self._phong_shading_with_background(self._background, depth_map_amplified, self._led_directions,
                                                     self._specular_components, self._diffuse_components, self._alpha,
                                                     self._kd, self._ks)

        # Darkening mask: The closer the object is to the camera, the darker it appears
        result -= (ncp.ones_like(result) - depth_map[:, :, ncp.newaxis]) * self._darkening_mask_strength

        # TODO: Elastometer displacement (see GelSight simulation paper) missing

        result = ncp.clip(result, 0.0, 1.0)
        if cupy_used:
            return ncp.asnumpy(result)
        else:
            return result

    def _get_requested_type(self) -> ObjectType:
        # The object type is Shape since the base of the model is the black cuboid
        return ObjectType.SHAPE

    @property
    def resolution(self) -> Tuple[int, int]:
        return tuple(self._resolution)  # type: Tuple[int, int]
