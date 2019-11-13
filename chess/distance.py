import numpy as np
from sklearn.metrics import pairwise_distances

from chess import globals


def check_input_array(x):
    if not isinstance(x, np.ndarray):
        raise TypeError(f'Expected type np.ndarray. Got {type(x)} instead.')
    if x.ndim != 2:
        raise ValueError(f'Expected array to have 2 dimensions. Got {x.ndim} instead.')
    if x.shape[0] == 0:
        raise ValueError(f'Expected array to have at least one point.')
    if x.shape[1] == 0:
        raise ValueError(f'Expected array to have points with non-zero dimensions.')
    return True


def calculate_distances(
        x, y,
        metric: str,
        count_calls: bool = False,
        use_tensorflow: bool = False
) -> np.ndarray:
    f"""
    Calculates the pairwise distances between the points in x and y using the metric specified.
    Optionally counts the number of distance calculations and updates the DF_CALLS variable in globals.
    
    :param x: 2-d array of points.
    :param y: 2-d array of points.
    :param metric: distance metric from scikit-learn pairwise to use.
    :param count_calls: Weather or not to count the number of distance calculations made.
    :param use_tensorflow: Weather or not to use tensorflow to compute distances.
    
    :return: numpy array of pairwise distances.
    """

    check_input_array(x)
    check_input_array(y)

    if count_calls:
        globals.DF_CALLS += x.shape[0] * y.shape[0]

    if use_tensorflow:
        raise NotImplementedError(f'Tensorflow distances have not yet been ported from old repo.')
    else:
        distances = pairwise_distances(x, y, metric)
        return np.maximum(distances, np.zeros_like(distances, dtype=globals.RADII_DTYPE))
