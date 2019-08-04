import os
from time import time

import numpy as np

import config
from src.search import Search

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_data(filename: str, num_rows: int, num_dims: int) -> np.memmap:
    """ Read data from memmap on disk.

    :param filename: filename to read.
    :param num_rows: number of rows in memmap.
    :param num_dims: number of columns in memmap.
    :return: numpy.memmap object.
    """
    return np.memmap(
        filename=filename,
        dtype='float32',
        mode='r',
        shape=(num_rows, num_dims),
    )


def make_clusters(distance_function: str, clustering_depth: int, filename: str) -> Search:
    """ Make new search object.

    :param distance_function: distance function to use.
    :param clustering_depth: maximum clustering depth to go to.
    :param filename: Name of csv file in which to store how long clustering took.
    :return: search object that was created.
    """

    data: np.memmap = read_data(config.DATA_FILE, config.NUM_ROWS - 10_000, config.NUM_DIMS)
    config.MAX_DEPTH = clustering_depth

    start = time()
    search_object = Search(data=data, distance_function=distance_function)
    end = time()

    with open(filename, 'a') as outfile:
        outfile.write(f'{clustering_depth},{end - start:.6f}\n')

    names_file = f'logs/names_{distance_function}_{clustering_depth}_{config.LFD_LIMIT}.csv'
    search_object.print_names(filename=names_file)

    info_file = f'logs/info_{distance_function}_{clustering_depth}_{config.LFD_LIMIT}.csv'
    search_object.print_info(filename=info_file)

    return search_object


def read_clusters(distance_function: str, clustering_depth: int) -> Search:
    """ read search object from csv files.

    :param distance_function: distance function that was used for the clustering.
    :param clustering_depth: maximum depth of the clustering.
    :return: search object that was read.
    """

    data: np.memmap = read_data(config.DATA_FILE, config.NUM_ROWS - 10_000, config.NUM_DIMS)
    config.MAX_DEPTH = clustering_depth

    return Search(
        data=data,
        distance_function=distance_function,
        reading=True,
        names_file=f'logs/names_{distance_function}_{clustering_depth}_{config.LFD_LIMIT}.csv',
        info_file=f'logs/info_{distance_function}_{clustering_depth}_{config.LFD_LIMIT}.csv',
    )


def search(search_object: Search, radius: float, filename: str) -> None:
    """ Perform iteratively deepening search and store results in a csv file.

    :param search_object: Search object to search in.
    :param radius: radius to use for search.
    :param filename: name of csv to which to write search results.
    :return:
    """

    samples = read_data(config.SAMPLES_FILE, 10_000, config.NUM_DIMS)

    for sample in samples:

        start = time()
        linear_results = search_object.linear_search(sample, radius)
        one = time() - start

        for search_depth in range(4, 20):
            start = time()
            clustered_results = search_object.clustered_search(sample, radius, search_depth)
            two = time() - start

            clustered_success = set(linear_results) == set(clustered_results)

            with open(filename, 'a') as outfile:
                outfile.write(f'{clustered_success},{radius},{one:.6f},{two:.6f},{search_depth}\n')
    return


if __name__ == '__main__':
    np.random.seed(1234)

    times_file = f'logs/times.csv'
    if not os.path.exists(times_file):
        with open(times_file, 'w') as outfile_:
            outfile_.write(f'depth,time\n')

    for d in [4, 5, 6, 7, 8, 9, 10, 15, 20]:
        make_clusters(distance_function='l2', clustering_depth=d, filename=times_file)

    search_times = f'logs/searches.csv'
    if not os.path.exists(search_times):
        with open(search_times, 'w') as outfile_:
            outfile_.write('clustered_success,radius,linear_time,clustered_time,search_depth\n')

    search_object_ = read_clusters(distance_function='l2', clustering_depth=20)

    for r in [25_000, 50_000, 100_000]:
        search(search_object=search_object_, radius=r, filename=search_times)
