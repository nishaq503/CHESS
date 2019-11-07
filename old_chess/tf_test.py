import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    print(tf.__version__)
    print(tf.test.is_gpu_available())
