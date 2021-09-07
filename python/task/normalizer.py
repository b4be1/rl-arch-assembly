import numpy as np


class Normalizer:
    def __init__(self, limits_lower: np.ndarray, limits_upper: np.ndarray):
        assert limits_lower.shape == limits_upper.shape, \
            "The shape of the lower limits and the upper limits must be equal, but received {} and {}".format(
                limits_lower, limits_upper)
        self.__space_center = 0.5 * (limits_upper + limits_lower)
        self.__space_width = limits_upper - limits_lower

    def normalize(self, unnormalized: np.ndarray):
        return (unnormalized - self.__space_center) / (self.__space_width / 2)

    def denormalize(self, normalized: np.ndarray):
        return normalized * (self.__space_width / 2) + self.__space_center
