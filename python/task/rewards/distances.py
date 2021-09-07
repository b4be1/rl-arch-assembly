from typing import Union

import numpy as np


def ssd_log_distance(
        distance: Union[np.ndarray, float], logarithm_weight: float = 0.01,
        logarithm_epsilon: float = 1e-5) -> np.ndarray:
    """
    A distance function that consists of a quadratic term that penalizes large distances and a logarithm that ensures
    that small distances are also penalized sufficiently.
    Adapted from: Levine et al., 2015: "Learning Contact-Rich Manipulation Skills with Guided Policy Search"

    :param distance:            the distance to the target value
    :param logarithm_weight:    the weight of the logarithmic term
    :param logarithm_epsilon:   the epsilon in the logarithmic term that ensures the logarithm is always defined
    :return:                    the distance
    """
    return distance + logarithm_weight * (np.log(distance + logarithm_epsilon) - np.log(logarithm_epsilon))
