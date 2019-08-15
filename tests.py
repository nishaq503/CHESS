import os

import numpy as np

from src.utils import tf_calculate_distance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def check_cosine():
    x = np.asarray([[1, 2, 3], [2, 3, 4]])
    y = np.asarray([3, 6, 9])
    distances = tf_calculate_distance(x, y, 'cos')
    distances = np.asarray(distances)
    print(distances)
    return


if __name__ == '__main__':
    check_cosine()
