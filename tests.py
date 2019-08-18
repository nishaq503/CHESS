import os

import numpy as np

import config
from src.utils import numpy_calculate_distance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def check_hamming(df: str, batch: bool = False):
    a = np.asarray([1, 2, 3], dtype=np.int8)
    distances = numpy_calculate_distance(a, a, df)
    print(distances)

    return


if __name__ == '__main__':
    # check_df('l2')
    check_hamming('hamming')
    # check_hamming('hamming', batch=True)
