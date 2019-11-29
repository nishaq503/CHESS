import os
from time import time

import numpy as np
from chess import CHESS


def benchmark_clustering(chess_object: CHESS, timing_file: str, staring_depth: int, ending_depth: int) -> CHESS:
    for depth in range(staring_depth, ending_depth + 1):
        start = time()
        chess_object.deepen(levels=1)
        end = time()
        with open(timing_file, 'a') as outfile:
            outfile.write(f'{depth}, {end - start:.5f}\n')
        print(f'at depth {depth}, we had {len(list(chess_object.root.leaves()))} leaves.')

    return chess_object


if __name__ == '__main__':
    base_path = '/home/nishaq/APOGEE2/'
    data_file = base_path + 'apo25m_data.memmap'
    queries_file = base_path + 'apo25m_queries.memmap'

    num_data = 264_160 - 10_000
    num_dims = 8_575

    data_memmap = np.memmap(
        filename=data_file,
        dtype=np.float32,
        mode='r',
        shape=(num_data, num_dims),
    )
    queries_memmap = np.memmap(
        filename=queries_file,
        dtype=np.float32,
        mode='r',
        shape=(10_000, num_dims),
    )

    clustering_benchmarks_file = 'logs/apogee2_clustering_times.csv'
    if not os.path.exists(clustering_benchmarks_file):
        with open(clustering_benchmarks_file, 'w') as of:
            of.write('new_depth, time_taken\n')

    co = CHESS(
        data=data_memmap,
        metric='euclidean',
        max_depth=0,
        min_points=10,
    )

    old_depth = 15
    s = time()
    co.load(filename=f'logs/chess_apogee2_{old_depth}.csv')
    e = time()
    print(f'loading chess object of depth {old_depth} took {e - s:.5f} seconds.')
    for d in range(old_depth, 100, 5):
        co = benchmark_clustering(
            chess_object=co,
            timing_file=clustering_benchmarks_file,
            staring_depth=d + 1,
            ending_depth=d + 5,
        )
        co.write(filename=f'logs/chess_apogee2_{d + 5}.csv')
