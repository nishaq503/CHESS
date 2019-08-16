import os
import numpy as np
from src.utils import tf_calculate_distance, numpy_calculate_distance, tf_calculate_pairwise_distances

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def check_df(df: str, batch: bool = False):
    x = np.asarray([3, 6, 9])
    y = np.asarray([[1, 2, 3], [2, 3, 4], [5, 5, 5]])

    print('x:\n', x)
    print('y:\n', y)

    if batch:
        tf_distances = tf_calculate_pairwise_distances(y, df)
        np_distances = numpy_calculate_distance(y, y, df)
    else:
        tf_distances = tf_calculate_distance(x, y, df)
        np_distances = numpy_calculate_distance(x, y, df)
    tf_distances = np.asarray(tf_distances)

    print('tf:\n', tf_distances)
    print('np:\n', np_distances)

    return


if __name__ == '__main__':
    # check_df('l2')
    # check_df('cos')
    check_df('cos', batch=True)
