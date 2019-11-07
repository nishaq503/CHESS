from time import time

from src.search import Search, get_data_and_queries
from src import globals


def make_clusters(
        dataset: str,
        metric: str,
        depth: int,
        timing_filename: str = None,
) -> Search:
    """
    Makes a new search-object with a cluster-tree of the requested depth.

    :param dataset: dataset to cluster.
    :param metric: distance metric to use for clustering.
    :param depth: maximum depth of cluster tree.
    :param timing_filename: optional .csv file in which to store the time it took for clustering.
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

    if timing_filename is not None:
        with open(timing_filename, 'a') as outfile:
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
        for i in range(old_depth, new_depth):
            globals.MAX_DEPTH = i + 1
            start = time()
            search_object.cluster_deeper(new_depth=i + 1)
            end = time()

            if timing_filename is not None:
                with open(timing_filename, 'a') as outfile:
                    outfile.write(f'{search_object.dataset},{search_object.metric},{i},{i + 1},{end - start:.6f}\n')

            print_summary(depth=i + 1, names=names_file, info=info_file)

    else:
        globals.MAX_DEPTH = new_depth
        search_object.cluster_deeper(new_depth=new_depth)
        print_summary(depth=new_depth, names=names_file, info=info_file)

    return search_object
