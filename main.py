import os
from time import time

import numpy as np

import config
from src.search import Search

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_data(filename: str, num_rows: int, num_dims: int) -> np.memmap:
    return np.memmap(
        filename=filename,
        dtype='float32',
        mode='r',
        shape=(num_rows, num_dims),
    )


def make_clusters(distance_function: str, max_depth: int):
    data: np.memmap = read_data(config.DATA_FILE, config.NUM_ROWS - 10_000, config.NUM_DIMS)
    config.MAX_DEPTH = max_depth

    search_object = Search(data=data, distance_function=distance_function)

    names_file = f'logs/names_{distance_function}_{max_depth}_{config.LFD_LIMIT}.csv'
    search_object.print_names(filename=names_file)

    info_file = f'logs/info_{distance_function}_{max_depth}_{config.LFD_LIMIT}.csv'
    search_object.print_info(filename=info_file)

    return search_object


def read_clusters(distance_function: str, max_depth: int):
    data: np.memmap = read_data(config.DATA_FILE, config.NUM_ROWS - 10_000, config.NUM_DIMS)
    config.MAX_DEPTH = max_depth

    return Search(
        data=data,
        distance_function=distance_function,
        reading=True,
        names_file=f'logs/names_{distance_function}_{max_depth}_{config.LFD_LIMIT}.csv',
        info_file=f'logs/info_{distance_function}_{max_depth}_{config.LFD_LIMIT}.csv',
    )


def search(clustering_depth, search_depth):
    search_object = make_clusters(distance_function='l2', max_depth=clustering_depth)
    read_object = read_clusters(distance_function='l2', max_depth=clustering_depth)

    samples = read_data(config.SAMPLES_FILE, 10_000, config.NUM_DIMS)
    radius = 50_000

    print('clustered_success,read_success,linear_time,clustered_time,read_time')
    for sample in samples:
        zero = time()

        results = search_object.linear_search(sample, radius)
        one = time() - zero

        clustered_results = search_object.clustered_search(sample, radius, search_depth)
        two = time() - zero - one

        read_results = read_object.clustered_search(sample, radius, search_depth)
        three = time() - zero - one - two

        clustered_success = set(results) == set(clustered_results)
        read_success = set(results) == set(read_results)

        print(f'{clustered_success},{read_success},{one:.6f},{two:.6f},{three:.6f}')

        break
    return


if __name__ == '__main__':
    np.random.seed(1234)
    search(5, 5)
