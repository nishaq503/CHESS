import os
from typing import Dict

import numpy as np

from src import globals
from src.benchmarks import make_clusters, read_clusters, deepen_clustering, benchmark_search
from src.search import get_data_and_queries

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(
        dataset: str,
        metric: str,
        initial_depth: int = 1,
        final_depth: int = 100,
        do_initial_clustering: bool = True,
        clustering_times_filename: str = f'logs/clustering_times.csv',
        run_search_benchmarks: bool = True,
        search_benchmarks_filename: str = f'logs/search_benchmarks',
        radii: Dict = globals.SEARCH_RADII
):
    globals.MAX_DEPTH = initial_depth

    if dataset == 'GreenGenes':
        globals.MIN_POINTS = 10
        globals.MIN_RADIUS = 2.0 / globals.GREENGENES_NUM_DIMS

    if not os.path.exists(clustering_times_filename):
        with open(clustering_times_filename, 'a') as outfile:
            outfile.write(f'dataset,metric,starting_depth,ending_depth,time_taken(s)\n')

    search_object = make_clusters(
        dataset=dataset,
        metric=metric,
        depth=initial_depth,
        clustering_times_filename=clustering_times_filename,
    ) if do_initial_clustering else read_clusters(
        dataset=dataset,
        metric=metric,
        depth=initial_depth,
    )

    if final_depth > initial_depth:
        search_object = deepen_clustering(
            search_object=search_object,
            old_depth=initial_depth,
            new_depth=final_depth,
            iterative=True,
            timing_filename=clustering_times_filename,
        )

    if run_search_benchmarks:
        search_benchmarks_filename = f'logs/{search_benchmarks_filename}_{dataset}_{metric}.csv'
        if not os.path.exists(search_benchmarks_filename):
            with open(search_benchmarks_filename, 'w') as outfile:
                outfile.write(f'depth,radius,correctness,false_negative_rate,num_hits,num_clusters_searched,'
                              f'fraction_searched,df_calls_made,linear_time,chess_time,speedup_factor\n')

        _, queries = get_data_and_queries(dataset)
        for radius in list(map(globals.RADII_DTYPE, radii[metric])):
            benchmark_search(
                search_object=search_object,
                queries=queries,
                num_queries=10,
                radius=radius,
                search_benchmarks_filename=search_benchmarks_filename,
            )
    return


if __name__ == '__main__':
    np.random.seed(42)

    # from argparse import ArgumentParser
    #
    # parser = ArgumentParser('CHESS')
    # parser.add_argument('dataset', choices={'APOGEE', 'GreenGenes'})
    # parser.add_argument('metric', choices=globals.DISTANCE_FUNCTIONS)
    # parser.add_argument('--initial-depth', nargs=1, default=1)
    # parser.add_argument('--final-depth', nargs=1, default=100)
    # parser.add_argument('--do-initial-clustering', action='store_true')
    # parser.add_argument('--run-search-benchmarks', action='store_true')
    # args = parser.parse_args()
    # main(**vars(args))

    # main(
    #     dataset='APOGEE',
    #     metric='euclidean',
    #     initial_depth=1,
    #     final_depth=100,
    #     do_initial_clustering=True,
    #     run_search_benchmarks=False,
    # )
    # main(
    #     dataset='APOGEE',
    #     metric='cosine',
    #     initial_depth=1,
    #     final_depth=100,
    #     do_initial_clustering=True,
    #     run_search_benchmarks=False
    # )
    main(
        dataset='GreenGenes',
        metric='hamming',
        initial_depth=50,
        final_depth=50,
        do_initial_clustering=False,
        run_search_benchmarks=True
    )
