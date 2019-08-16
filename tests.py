import os
import numpy as np
from src.utils import tf_calculate_distance, numpy_calculate_distance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def check_df(df):
    x = np.asarray([[1, 2, 3], [2, 3, 4]])
    y = np.asarray([3, 6, 9])

    tf_distances = tf_calculate_distance(x, y, df)
    tf_distances = np.asarray(tf_distances)
    np_distances = numpy_calculate_distance(x, y, df)

    print('tf:\n', tf_distances)
    print('np:\n', np_distances)

    return


if __name__ == '__main__':
    check_df('l2')
    check_df('cos')
