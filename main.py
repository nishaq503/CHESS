import os

import numpy as np

from src import globals
from src.benchmarks import make_clusters, read_clusters, deepen_clustering

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    np.random.seed(42)

    initial_depth = 1
    globals.MAX_DEPTH = initial_depth

    dataset = 'APOGEE'
    metric = 'euclidean'

    timing_filename = f'logs/clustering_times.csv'
    if not os.path.exists(timing_filename):
        with open(timing_filename, 'a') as outfile:
            outfile.write(f'dataset,metric,starting_depth,ending_depth,time_taken(s)\n')

    make_clusters(
        dataset=dataset,
        metric=metric,
        depth=initial_depth,
        timing_filename=timing_filename,
    )

    search_object = read_clusters(
        dataset=dataset,
        metric=metric,
        depth=initial_depth,
    )

    max_depth = 5
    search_object = deepen_clustering(
        search_object=search_object,
        old_depth=initial_depth,
        new_depth=max_depth,
        iterative=True,
        timing_filename=timing_filename,
    )
