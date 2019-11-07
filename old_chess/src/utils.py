from time import time

import numpy as np
import tensorflow as tf

import old_chess.config as config


def tf_l2_norm(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1), 0.0))


def numpy_l2_norm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(np.sqrt(np.einsum('ij,ij->i', x, x)[:, None]
                              + np.einsum('ij,ij->i', y, y)
                              - 2 * np.dot(x, y.T)),
                      0.0)


def batch_tf_l2_norm(x: tf.Tensor) -> tf.Tensor:
    x_sq = tf.reduce_sum(tf.square(x), axis=1)
    xx, yy = tf.reshape(x_sq, shape=[-1, 1]), tf.reshape(x_sq, shape=[1, -1])
    return tf.sqrt(tf.maximum(xx + yy - 2 * tf.matmul(x, x, transpose_b=True), 0.0))


def tf_cosine_distance(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x_normalized = tf.nn.l2_normalize(x, axis=1)
    y_normalized = tf.nn.l2_normalize(y, axis=1)
    product = tf.matmul(x_normalized, y_normalized, transpose_b=True)
    return tf.maximum(1 - product, 1e-16)


def numpy_cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_normalized = x.T / np.linalg.norm(x, axis=1)
    y_normalized = y.T / np.linalg.norm(y, axis=1)
    product = np.matmul(x_normalized.T, y_normalized)
    return np.maximum(1 - product, 1e-16)


def batch_hamming(x: np.ndarray) -> np.ndarray:
    return (x[:, None, :] != x).sum(2)


def hamming(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x != y[:, None, :]).sum(2)


def tf_calculate_distance(a: np.ndarray, b: np.ndarray, df: str, logfile: str = None, d: int = None) -> np.ndarray:
    """ Calculates the distance between a and b using the distance function requested.

    :param a: numpy array of points.
    :param b: numpy array of points.
    :param df: 'l2', 'cos', or 'emd'.
    :param logfile: .csv to write logs in.
    :param d: cluster_depth for logfile.
    :return: pairwise distances between points in a and b.
    """

    if logfile:
        with open(logfile, 'a') as outfile:
            outfile.write(f'tf_calculate_distance,{d},{df},start,{time():.8f}\n')

    if df == 'l2':
        distance = tf_l2_norm
    elif df == 'cos':
        distance = tf_cosine_distance
    elif df == 'hamming':
        distance = hamming
    else:
        raise ValueError('Invalid distance function given. Must be \'l2\', \'cos\', or \'hamming\'.')

    x = tf.placeholder(dtype=tf.float64, shape=[None, None])
    y = tf.placeholder(dtype=tf.float64, shape=[None, None])

    a, b = np.asarray(a), np.asarray(b)
    squeeze_a, squeeze_b = False, False
    if a.ndim == 1:
        a = np.expand_dims(a, 0)
        squeeze_a = True
    if b.ndim == 1:
        b = np.expand_dims(b, 0)
        squeeze_b = True

    config.DF_CALLS += np.shape(a)[0] * np.shape(b)[0]

    if df in ['l2', 'cos']:
        with tf.Session() as sess:
            [distances] = sess.run([distance(x, y)], feed_dict={x: a, y: b})
    else:
        distances = distance(a, b)

    distances = np.asarray(distances)
    if df == 'cos':
        if squeeze_a and squeeze_b:
            return distances[0]
        elif squeeze_a:
            return distances[0]
        elif squeeze_b:
            return distances.T[0]
    elif df == 'hamming':
        if squeeze_a and squeeze_b:
            return distances[0]
        elif squeeze_a:
            return distances.T[0]
        elif squeeze_b:
            return distances[0]

    if logfile:
        with open(logfile, 'a') as outfile:
            outfile.write(f'tf_calculate_distance,{d},{df},end,{time():.8f}\n')

    return distances


def tf_calculate_pairwise_distances(a: np.ndarray, df: str, logfile: str = None, d: int = None) -> np.ndarray:
    """ Calculates the pairwise distance between all elements of a using the distance function requested.

    :param a: numpy array of points.
    :param df: 'l2', 'cos', or 'emd'.
    :param logfile: .csv to write logs in.
    :param d: cluster_depth for logfile.
    :return: pairwise distances between points in a.
    """

    if logfile:
        with open(logfile, 'a') as outfile:
            outfile.write(f'tf_calculate_pairwise_distances,{d},{df},start,{time():.8f}\n')

    if df == 'l2':
        distance = batch_tf_l2_norm
    elif df == 'cos':
        def distance(x_): return tf_cosine_distance(x_, x_)
    elif df == 'hamming':
        distance = batch_hamming
    else:
        raise ValueError('Invalid distance function given. Must be \'l2\', \'cos\', or \'hamming\'.')

    a = np.asarray(a)
    if a.ndim == 1:
        a = np.expand_dims(a, 0)

    config.DF_CALLS += np.shape(a)[0] * np.shape(a)[0]

    x = tf.placeholder(dtype=tf.float64, shape=[None, None])

    if df in ['l2', 'cos']:
        with tf.Session() as sess:
            [distances] = sess.run([distance(x)], feed_dict={x: a})
    else:
        distances = distance(a)

    if logfile:
        with open(logfile, 'a') as outfile:
            outfile.write(f'tf_calculate_pairwise_distances,{d},{df},end,{time():.8f}\n')

    return np.asarray(distances)


def numpy_calculate_distance(a: np.array, b: np.array, df: str, logfile: str = None, d: int = None) -> np.array:
    """ Calculates the distance between a and b using the distance function requested with numpy.

        :param a: numpy array of points.
        :param b: numpy array of points.
        :param df: 'l2', 'cos', or 'emd'.
        :param logfile: .csv to write logs in.
        :param d: cluster_depth for logfile.
        :return: pairwise distances between points in a and b.
        """

    if logfile:
        with open(logfile, 'a') as outfile:
            outfile.write(f'numpy_calculate_distance,{d},{df},start,{time():.8f}\n')

    if df == 'l2':
        distance = numpy_l2_norm
    elif df == 'cos':
        distance = numpy_cosine_distance
    elif df == 'hamming':
        distance = hamming
    else:
        raise ValueError('Invalid distance function given. Must be \'l2\', \'cos\', or \'hamming\'.')

    a, b = np.array(a, dtype=np.float128), np.array(b, dtype=np.float128)
    squeeze_a, squeeze_b = False, False
    if a.ndim == 1:
        a = np.expand_dims(a, 0)
        squeeze_a = True
    if b.ndim == 1:
        b = np.expand_dims(b, 0)
        squeeze_b = True

    config.DF_CALLS += np.shape(a)[0] * np.shape(b)[0]
    try:
        distances = distance(a, b)
    except:
        raise ValueError(f'had a problem with arrays {a.shape} and {b.shape}.\na:\n{a}\n b:\n{b}')

    if logfile:
        with open(logfile, 'a') as outfile:
            outfile.write(f'numpy_calculate_distance,{d},{df},end,{time():.8f}\n')

    if df == 'hamming':
        if squeeze_a and squeeze_b:
            return distances[0]
        elif squeeze_a:
            return distances.T[0]
        elif squeeze_b:
            return distances[0]
    else:
        if squeeze_a and squeeze_b:
            return distances[0]
        elif squeeze_a:
            return distances[0]
        elif squeeze_b:
            return distances.T[0]

    return distances
