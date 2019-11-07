import os

import numpy as np

import config
from src.utils import numpy_calculate_distance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def calculate_center():
    """ Calculate the geometric median of the potential centers. This is the center of the cluster.

    :return: index of center of cluster.
    """
    pairwise_distances = np.asarray([0, 1, 2])
    print(pairwise_distances.ndim)
    return np.sum(pairwise_distances)


if __name__ == '__main__':
    print(calculate_center())
