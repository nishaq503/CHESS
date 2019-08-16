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
        outfile.write(f'{clustering_depth},{end - start:.6f},{distance_function}\n')

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


def benchmark_search(search_object: Search, radius: float, filename: str) -> None:
    """ Perform iteratively deepening search and store results in a csv file.

    :param search_object: Search object to search in.
    :param radius: radius to use for search.
    :param filename: name of csv to which to write search results.
    :return:
    """
    samples = read_data(config.SAMPLES_FILE, 10_000, config.NUM_DIMS)
    number_searched = 0

    for sample in samples:

        start = time()
        linear_results = search_object.linear_search(sample, radius)
        one = time() - start

        for search_depth in range(4, 21):
            config.DF_CALLS = 0
            start = time()
            clustered_results, num_clusters = search_object.clustered_search(sample, radius, search_depth)
            two = time() - start
            success = set(linear_results) == set(clustered_results)
            with open(filename, 'a') as outfile:
                outfile.write(f'{success},{radius},{search_depth},{len(clustered_results)},{num_clusters},'
                              f'{one:.6f},{two:.6f},{config.DF_CALLS}\n')
                outfile.flush()
        number_searched += 1
        if number_searched >= 29:
            break
    return


if __name__ == '__main__':
    np.random.seed(1234)
    distance_function_ = 'cos'
    clustering_depth_ = 50

    times_file = f'logs/times.csv'
    if not os.path.exists(times_file):
        with open(times_file, 'w') as outfile_:
            outfile_.write(f'depth,time,distance_function\n')

    for d in [30]:  # [4, 5, 6, 7, 8, 9, 10, 15, 20]:
        make_clusters(distance_function=distance_function_, clustering_depth=d, filename=times_file)
        break

    search_object_ = read_clusters(distance_function=distance_function_, clustering_depth=clustering_depth_)

    # metadata_filename = f'compressed/encoding_metadata_{distance_function_}_{clustering_depth_}.pickle'
    # integer_filename = f'compressed/integer_encodings_{distance_function_}_{clustering_depth_}'
    # integer_zip = f'compressed/integer_encodings_{distance_function_}_{clustering_depth_}.zip'
    # search_object_.compress(metadata_filename, integer_filename, integer_zip)

    # search_results = f'logs/searches_{distance_function_}_{clustering_depth_}.csv'
    # if not os.path.exists(search_results):
    #     with open(search_results, 'w') as outfile_:
    #         outfile_.write('success,radius,search_depth,output_size,clusters_searched,'
    #                        'linear_time,clustered_time,df_calls\n')
    #
    # search_times = f'logs/search_times_{distance_function_}_{clustering_depth_}.csv'
    # for r in [i * (10**-3) for i in [1, 5, 10]]:
    #     benchmark_search(search_object_, r, search_results)

