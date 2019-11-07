import os

import numpy as np

from src import globals
from src.benchmarks import make_clusters, read_clusters, deepen_clustering, benchmark_search
from src.search import get_data_and_queries

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    np.random.seed(42)

    initial_depth = 5
    globals.MAX_DEPTH = initial_depth

    dataset = 'APOGEE'
    metric = 'euclidean'

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

    # max_depth = 5
    # search_object = deepen_clustering(
    #     search_object=search_object,
    #     old_depth=initial_depth,
    #     new_depth=max_depth,
    #     iterative=True,
    #     timing_filename=timing_filename,
    # )
    # (f'{depth},{radius},{correctness},{false_negative_rate},{len(results)},{num_clusters},'
    # f'{fraction_searched},{globals.DF_CALLS},{linear_time},{chess_time},{speedup_factor}\n')

    search_benchmarks_filename = f'logs/search_benchmarks_{dataset}_{metric}.csv'
    if not os.path.exists(search_benchmarks_filename):
        with open(search_benchmarks_filename, 'w') as outfile:
            outfile.write(f'depth,radius,correctness,false_negative_rate,num_hits,num_clusters_searched,'
                          f'fraction_searched,df_calls_made,linear_time,chess_time,speedup_factor\n')

    greengenes_min_radius = int(0.01 * globals.GREENGENES_NUM_DIMS)
    radii = {
        'euclidean': [2000, 4000],
        'cosine': [0.0025, 0.005, 0.01],
        'hamming': [greengenes_min_radius, 2 * greengenes_min_radius, 5 * greengenes_min_radius]
    }

    _, queries = get_data_and_queries(dataset)

    for radius in radii[metric]:
        benchmark_search(
            search_object=search_object,
            queries=queries,
            num_queries=5,
            radius=radius,
            search_benchmarks_filename=search_benchmarks_filename,
        )
