import os
from time import time

import numpy as np

from chess import CHESS


# noinspection DuplicatedCode
def benchmark_clustering(chess_object: CHESS, timing_file: str, staring_depth: int, ending_depth: int) -> CHESS:
    for depth in range(staring_depth, ending_depth + 1):
        start = time()
        chess_object.deepen(levels=1)
        end = time()
        with open(timing_file, 'a') as outfile:
            outfile.write(f'{depth}, {len(list(chess_object.root.leaves()))}, {end - start:.5f}\n')

    return chess_object


# noinspection DuplicatedCode
def benchmark_apogee2_100k():
    base_path = '/data/nishaq/APOGEE2/'
    data_file = base_path + 'apo100k_data.memmap'

    num_data = 100_000
    num_dims = 8_575

    data_memmap = np.memmap(
        filename=data_file,
        dtype=np.float32,
        mode='r',
        shape=(num_data, num_dims),
    )

    clustering_benchmarks_file = '../benchmarks/logs/apo100k_clustering_times.csv'
    if not os.path.exists(clustering_benchmarks_file):
        with open(clustering_benchmarks_file, 'w') as of:
            of.write('new_depth, num_leaves, time_taken\n')

    np.random.seed(42)
    co = CHESS(
        data=data_memmap,
        metric='euclidean',
        max_depth=0,
        min_points=10,
    )
    old_depth = 0
    if old_depth > 0:
        s = time()
        co = co.load(filename=f'../benchmarks/logs/chess_apo100k_{old_depth}.json', data=data_memmap)
        print(f'reading from json with depth {old_depth} took {time() - s:.4f} seconds.')
    max_depth, step = 50, 10
    for d in range(old_depth, max_depth, step):
        co = benchmark_clustering(
            chess_object=co,
            timing_file=clustering_benchmarks_file,
            staring_depth=d + 1,
            ending_depth=d + step,
        )
        co.write(filename=f'../benchmarks/logs/chess_apo100k_{d + step}.json')
    return


# noinspection DuplicatedCode
def benchmark_greengenes_100k():
    base_path = '/data/nishaq/GreenGenes/'
    data_file = base_path + 'gg100k_data.memmap'

    num_data = 100_000
    num_dims = 7_682

    data_memmap = np.memmap(
        filename=data_file,
        dtype=np.int8,
        mode='r',
        shape=(num_data, num_dims),
    )

    clustering_benchmarks_file = '../benchmarks/logs/gg100k_clustering_times.csv'
    if not os.path.exists(clustering_benchmarks_file):
        with open(clustering_benchmarks_file, 'w') as of:
            of.write('new_depth, num_leaves, time_taken\n')

    np.random.seed(42)
    co = CHESS(
        data=data_memmap,
        metric='hamming',
        max_depth=0,
        min_points=10,
    )
    old_depth = 0
    if old_depth > 0:
        s = time()
        co = co.load(filename=f'../benchmarks/logs/chess_gg100k_{old_depth}.json', data=data_memmap)
        print(f'reading from json with depth {old_depth} took {time() - s:.4f} seconds.')
    max_depth, step = 50, 10
    for d in range(old_depth, max_depth, step):
        co = benchmark_clustering(
            chess_object=co,
            timing_file=clustering_benchmarks_file,
            staring_depth=d + 1,
            ending_depth=d + step,
        )
        co.write(filename=f'../benchmarks/logs/chess_gg100k_{d + step}.json')
    return


if __name__ == '__main__':
    print('ready to benchmark!')
    benchmark_apogee2_100k()
    benchmark_greengenes_100k()
