import os
from time import time

import numpy as np

import config
from src.search import Search

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_data(filename: str, num_rows: int, num_dims: int, dtype) -> np.memmap:
    """ Read data from memmap on disk.

    :param filename: filename to read.
    :param num_rows: number of rows in memmap.
    :param num_dims: number of columns in memmap.
    :param dtype: data type of memmap.
    :return: numpy.memmap object.
    """
    return np.memmap(
        filename=filename,
        dtype=dtype,
        mode='r',
        shape=(num_rows, num_dims),
    )


def make_clusters(data: np.memmap, df: str, depth: int, filename: str) -> Search:
    """ Make new search object.

    :param data: data to cluster
    :param df: distance function to use.
    :param depth: maximum clustering depth to go to.
    :param filename: Name of csv file in which to store how long clustering took.
    :return: search object that was created.
    """
    start = time()
    search_object = Search(data=data, distance_function=df)
    end = time()

    with open(filename, 'a') as outfile:
        outfile.write(f'{depth},{end - start:.6f},{df}\n')

    names_file = f'logs/names_{df}_{depth}_{config.LFD_LIMIT}.csv'
    search_object.print_names(filename=names_file)

    info_file = f'logs/info_{df}_{depth}_{config.LFD_LIMIT}.csv'
    search_object.print_info(filename=info_file)

    return search_object


def read_clusters(data: np.memmap, df: str, depth: int) -> Search:
    """ read search object from csv files.

    :param data: data to cluster
    :param df: distance function that was used for the clustering.
    :param depth: maximum depth of the clustering.
    :return: search object that was read.
    """
    return Search(
        data=data,
        distance_function=df,
        reading=True,
        names_file=f'logs/names_{df}_{depth}_{config.LFD_LIMIT}.csv',
        info_file=f'logs/info_{df}_{depth}_{config.LFD_LIMIT}.csv',
    )


def benchmark_search(queries: np.memmap, search_object: Search, radius: float, filename: str) -> None:
    """ Perform iteratively deepening search and store results in a csv file.

    :param queries: queries to look for
    :param search_object: Search object to search in.
    :param radius: radius to use for search.
    :param filename: name of csv to which to write search results.
    :return:
    """
    number_searched = 0
    for sample in queries:
        start = time()
        linear_results = search_object.linear_search(sample, radius)
        one = time() - start

        for search_depth in range(4, config.MAX_DEPTH + 1, 1):
            config.DF_CALLS = 0
            start = time()
            results, num_clusters, fraction = search_object.clustered_search(sample, radius, search_depth)
            two = time() - start
            success = set(linear_results) == set(results)
            with open(filename, 'a') as outfile:
                outfile.write(f'{success},{radius},{search_depth},{len(results)},{num_clusters},'
                              f'{one:.6f},{two:.6f},{fraction:.6f},{config.DF_CALLS}\n')
                outfile.flush()
        number_searched += 1
        if number_searched >= 30:
            break
    return


if __name__ == '__main__':
    np.random.seed(1234)
    df_ = 'cos'
    depth_ = 20

    data_: np.memmap = read_data(filename=config.DATA_FILE,
                                 num_rows=config.NUM_ROWS - 10_000,
                                 num_dims=config.NUM_DIMS,
                                 dtype='float32')

    queries_: np.memmap = read_data(filename=config.SAMPLES_FILE,
                                    num_rows=10_000,
                                    num_dims=config.NUM_DIMS,
                                    dtype='float32')

    config.MAX_DEPTH = depth_

    # times_file = f'logs/times.csv'
    # if not os.path.exists(times_file):
    #     with open(times_file, 'w') as outfile_:
    #         outfile_.write(f'depth,time,distance_function\n')
    #
    # for d in [depth_]:  # [4, 5, 6, 7, 8, 9, 10, 15, 20]:
    #     make_clusters(data=data_, df=df_, depth=d, filename=times_file)
    #     break

    search_object_ = read_clusters(data=data_, df=df_, depth=depth_)

    # metadata_filename = f'compressed/encoding_metadata_{distance_function_}_{clustering_depth_}.pickle'
    # integer_filename = f'compressed/integer_encodings_{distance_function_}_{clustering_depth_}'
    # integer_zip = f'compressed/integer_encodings_{distance_function_}_{clustering_depth_}.zip'
    # search_object_.compress(metadata_filename, integer_filename, integer_zip)

    search_results = f'logs/searches_{df_}_{depth_}.csv'
    if not os.path.exists(search_results):
        with open(search_results, 'w') as outfile_:
            outfile_.write('success,radius,search_depth,output_size,clusters_searched,'
                           'linear_time,clustered_time,fraction_searched,df_calls\n')

    search_times = f'logs/search_times_{df_}_{depth_}.csv'
    # radii = [int(0.01 * config.SEQ_LEN), int(0.02 * config.SEQ_LEN), int(0.05 * config.SEQ_LEN)]
    # radii = [2000, 4000]
    radii = [0.0025, 0.005, 0.01]
    for r in radii:
        benchmark_search(queries=queries_, search_object=search_object_, radius=r, filename=search_results)
