import os

import numpy as np

from src import globals
from src.benchmarks import make_clusters, read_clusters, deepen_clustering, benchmark_search
from src.search import get_data_and_queries

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    np.random.seed(42)

    initial_depth = 30
    globals.MAX_DEPTH = initial_depth

    dataset = 'GreenGenes'
    metric = 'hamming'

    timing_filename = f'logs/clustering_times.csv'
    if not os.path.exists(timing_filename):
        with open(timing_filename, 'a') as outfile:
            outfile.write(f'dataset,metric,starting_depth,ending_depth,time_taken(s)\n')

    # make_clusters(
    #     dataset=dataset,
    #     metric=metric,
    #     depth=initial_depth,
    #     timing_filename=timing_filename,
    # )

    search_object = read_clusters(
        dataset=dataset,
        metric=metric,
        depth=initial_depth,
    )

    # max_depth = 50
    # search_object = deepen_clustering(
    #     search_object=search_object,
    #     old_depth=initial_depth,
    #     new_depth=max_depth,
    #     iterative=True,
    #     timing_filename=timing_filename,
    # )

    search_benchmarks_filename = f'logs/search_benchmarks_{dataset}_{metric}.csv'
    if not os.path.exists(search_benchmarks_filename):
        with open(search_benchmarks_filename, 'w') as outfile:
            outfile.write(f'depth,radius,correctness,false_negative_rate,num_hits,num_clusters_searched,'
                          f'fraction_searched,df_calls_made,linear_time,chess_time,speedup_factor\n')

    greengenes_min_radius = int(0.01 * globals.GREENGENES_NUM_DIMS)
    radii = {
        'euclidean': [2000, 4000],
        'cosine': [0.005, 0.001],
        'hamming': [0.01, 0.02, 0.05],
    }

    _, queries = get_data_and_queries(dataset)

    for radius in list(map(globals.RADII_DTYPE, radii[metric])):
        benchmark_search(
            search_object=search_object,
            queries=queries,
            num_queries=10,
            radius=radius,
            search_benchmarks_filename=search_benchmarks_filename,
        )
