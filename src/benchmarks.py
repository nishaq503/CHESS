from time import time
from typing import List

import numpy as np

from src import globals
from src.distance_functions import check_input_array
from src.search import Search, get_data_and_queries


def make_clusters(
        dataset: str,
        metric: str,
        depth: int,
        clustering_times_filename: str = None,
) -> Search:
    """
    Makes a new search-object with a cluster-tree of the requested depth.

    :param dataset: dataset to cluster.
    :param metric: distance metric to use for clustering.
    :param depth: maximum depth of cluster tree.
    :param clustering_times_filename: optional .csv file in which to store the time it took for clustering.
    :return: search object that was created.
    """
    start = time()
    search_object: Search = Search(
        dataset=dataset,
        metric=metric,
        names_file=f'logs/names_{metric}_{depth}.csv',
        info_file=f'logs/info_{metric}_{depth}.csv',
    )
    end = time()

    if clustering_times_filename is not None:
        with open(clustering_times_filename, 'a') as outfile:
            outfile.write(f'{dataset},{metric},{0},{depth},{end - start:.6f}\n')

    search_object.print_names()
    search_object.print_info()

    return search_object


def read_clusters(
        dataset: str,
        metric: str,
        depth: int,
        names_file: str = None,
        info_file: str = None,
) -> Search:
    """
    Read search-object from files.

    :param dataset: dataset to read.
    :param metric: metric that was used for clustering.
    :param depth: depth to which clustering was carried out.
    :param names_file: optional filepath of names_file to read from.
    :param info_file: optional filepath of info_file to read from.

    :return: search-object that was read in.
    """
    data, _ = get_data_and_queries(dataset=dataset)

    if names_file is None:
        names_file = f'logs/names_{metric}_{depth}.csv'
    if info_file is None:
        info_file = f'logs/info_{metric}_{depth}.csv'

    return Search(
        dataset=dataset,
        metric=metric,
        names_file=names_file,
        info_file=info_file,
        reading=True,
    )


def deepen_clustering(
        search_object: Search,
        old_depth: int,
        new_depth: int,
        iterative: bool = True,
        timing_filename: str = None,
        names_file: str = None,
        info_file: str = None,
) -> Search:
    def print_summary(depth: int, names: str, info: str):
        if names is None:
            names = f'logs/names_{search_object.metric}_{depth}.csv'
        if info is None:
            info = f'logs/info_{search_object.metric}_{depth}.csv'

        search_object.names_file = names
        search_object.print_names()

        search_object.info_file = info
        search_object.print_info()

    if iterative:
        for i in range(old_depth + 1, new_depth + 1):
            globals.MAX_DEPTH = i
            start = time()
            search_object.cluster_deeper(new_depth=i)
            end = time()

            if timing_filename is not None:
                with open(timing_filename, 'a') as outfile:
                    outfile.write(f'{search_object.dataset},{search_object.metric},{i - 1},{i},{end - start:.6f}\n')

            print_summary(depth=i, names=names_file, info=info_file)

    else:
        globals.MAX_DEPTH = new_depth
        search_object.cluster_deeper(new_depth=new_depth)
        print_summary(depth=new_depth, names=names_file, info=info_file)

    return search_object


def write_results(
        linear_results: List[int],
        results: List[int],
        linear_time: float,
        chess_time: float,
        search_benchmarks_filename: str,
        search_depth: int,
        radius: float,
        num_clusters: int,
        fraction_searched: float,
):
    correctness = (set(linear_results) == set(results)) and (len(linear_results) == len(results))
    if len(linear_results) > 0:
        false_negative_rate = 1 - (len(set(results)) / len(set(linear_results)))
    else:
        false_negative_rate = 0
    speedup_factor = linear_time / chess_time if chess_time > 0 else np.inf

    with open(search_benchmarks_filename, 'a') as outfile:
        outfile.write(f'{search_depth},{radius},{correctness},{false_negative_rate},{len(results)},{num_clusters},'
                      f'{fraction_searched},{globals.DF_CALLS},{linear_time},{chess_time},'
                      f'{speedup_factor:.3f}\n')
        outfile.flush()
    return


def benchmark_search(
        search_object: Search,
        queries: np.memmap,
        num_queries: int,
        radius: globals.RADII_DTYPE,
        search_benchmarks_filename: str,
):
    """
    Perform iteratively-deepening clustered-search and store benchmarks in a .csv file.

    :param search_object: search-object to run benchmarks on.
    :param queries: queries to search around.
    :param num_queries: number of queries to get benchmarks for.
    :param radius: search-radius to use for each query.
    :param search_benchmarks_filename: name of .csv file to write benchmarks to.
    """
    # max_depth = max(list(map(len, search_object.cluster_dict.keys())))
    # search_depths = list(range(0, max_depth + 1, 5))
    search_depths = [globals.MAX_DEPTH]
    search_queries = queries[:num_queries, :]

    for query in search_queries:
        query = np.expand_dims(query, 0)
        check_input_array(query)

        start = time()
        linear_results = search_object.linear_search(query=query, radius=radius)
        linear_time = time() - start

        for depth in search_depths:
            globals.DF_CALLS = 0

            start = time()
            results, num_clusters, fraction_searched = search_object.search(query, radius, depth, True)
            chess_time = time() - start

            write_results(
                linear_results=linear_results,
                results=results,
                linear_time=linear_time,
                chess_time=chess_time,
                search_benchmarks_filename=search_benchmarks_filename,
                search_depth=depth,
                radius=radius,
                num_clusters=num_clusters,
                fraction_searched=fraction_searched,
            )
    return


def benchmark_search_with_hits(
        search_object: Search,
        queries: np.memmap,
        num_queries: int,
        radius: globals.RADII_DTYPE,
        vs_falconn_filename: str,
        min_results: int = 1,
        max_results: int = 100,

):
    """
    Perform clustered-search. Store benchmarks Only if min_results <= num_hits <= max_results.

    :param search_object: search-object to run benchmarks on.
    :param queries: queries to search around.
    :param num_queries: number of queries to get benchmarks for.
    :param radius: search-radius to use for each query.
    :param vs_falconn_filename: name of .csv file to write benchmarks to.
    :param min_results: minimum number of results.
    :param max_results: maximum number of results.
    """
    search_depth = max(list(map(len, search_object.cluster_dict.keys())))
    num_searched: int = 0

    for q in queries:
        query = np.expand_dims(q, 0)
        check_input_array(query)
        globals.DF_CALLS = 0

        start = time()
        results, num_clusters, fraction_searched = search_object.search(query, radius, search_depth, True)
        chess_time = time() - start

        if not (min_results <= len(results) <= max_results):
            continue

        start = time()
        linear_results = search_object.linear_search(query=query, radius=radius)
        linear_time = time() - start

        write_results(
            linear_results=linear_results,
            results=results,
            linear_time=linear_time,
            chess_time=chess_time,
            search_benchmarks_filename=vs_falconn_filename,
            search_depth=search_depth,
            radius=radius,
            num_clusters=num_clusters,
            fraction_searched=fraction_searched,
        )
        num_searched += 1
        if num_searched > num_queries:
            break
    else:
        print(num_searched)
    return
