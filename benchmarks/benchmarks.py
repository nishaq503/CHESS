import os
from time import time

import numpy as np

from chess.criterion import *
from chess.manifold import Manifold

PLOT_NUMBER = 0


def benchmark_clustering(manifold: Manifold, timing_file: str, staring_depth: int, ending_depth: int) -> Manifold:
    for depth in range(staring_depth, ending_depth + 1):
        start = time()
        manifold.deepen(MaxDepth(depth), MinPoints(10))
        end = time()
        with open(timing_file, 'a') as outfile:
            outfile.write(f'{depth}, {len(manifold.graphs[-1])}, {end - start:.5f}\n')

    return manifold


# noinspection DuplicatedCode
def benchmark_apogee2_100k():
    base_path = '/scratch/nishaq/APOGEE2/'
    data_file = base_path + 'apo100k_data.memmap'

    num_data = 100_000
    num_dims = 8_575

    data_memmap: np.memmap = np.memmap(
        filename=data_file,
        dtype=np.float32,
        mode='r',
        shape=(num_data, num_dims),
    )

    clustering_benchmarks_file = 'manifold_logs/apogee2_100k_clustering_times.csv'
    if not os.path.exists(clustering_benchmarks_file):
        with open(clustering_benchmarks_file, 'w') as of:
            of.write('new_depth, num_leaves, time_taken\n')

    np.random.seed(42)
    manifold = Manifold(data_memmap, 'euclidean', propagate=False, calculate_neighbors=False)
    old_depth = 0
    if old_depth > 0:
        s = time()
        with open(f'manifold_logs/chess_apogee2_100k_{old_depth}.pickle', 'rb') as infile:
            manifold = manifold.pickle_load(fp=infile, data=data_memmap)
            manifold.__dict__['propagate'] = False
            manifold.__dict__['calculate_neighbors'] = False
        print(f'reading from json with depth {old_depth} took {time() - s:.4f} seconds.')
    step = 10
    for d in range(old_depth, 50, step):
        manifold = benchmark_clustering(
            manifold=manifold,
            timing_file=clustering_benchmarks_file,
            staring_depth=d + 1,
            ending_depth=d + step,
        )
        with open(f'manifold_logs/chess_apogee2_100k_{d + step}.pickle', 'wb') as outfile:
            manifold.pickle_dump(fp=outfile)
    return


# noinspection DuplicatedCode
def benchmark_greengenes_100k():
    base_path = '/scratch/nishaq/GreenGenes/'
    data_file = base_path + 'gg100k_data.memmap'

    num_data = 100_000
    num_dims = 7_682

    data_memmap = np.memmap(
        filename=data_file,
        dtype=np.int8,
        mode='r',
        shape=(num_data, num_dims),
    )

    clustering_benchmarks_file = 'manifold_logs/gg100k_clustering_times.csv'
    if not os.path.exists(clustering_benchmarks_file):
        with open(clustering_benchmarks_file, 'w') as of:
            of.write('new_depth, num_leaves, time_taken\n')

    np.random.seed(42)
    manifold = Manifold(data_memmap, 'hamming', propagate=False, calculate_neighbors=False)
    old_depth = 20
    if old_depth > 0:
        s = time()
        with open(f'manifold_logs/chess_gg100k_{old_depth}.pickle', 'rb') as infile:
            manifold = manifold.pickle_load(fp=infile, data=data_memmap)
            manifold.__dict__['propagate'] = False
            manifold.__dict__['calculate_neighbors'] = False
        print(f'reading from json with depth {old_depth} took {time() - s:.4f} seconds.')
    max_depth, step = 50, 10
    for d in range(old_depth, max_depth, step):
        manifold = benchmark_clustering(
            manifold=manifold,
            timing_file=clustering_benchmarks_file,
            staring_depth=d + 1,
            ending_depth=d + step,
        )
        with open(f'manifold_logs/chess_gg100k_{d + step}.pickle', 'wb') as outfile:
            manifold.pickle_dump(fp=outfile)
    return


# noinspection DuplicatedCode
def benchmark_search_gg100k():
    base_path = '/scratch/nishaq/GreenGenes/'
    data_file = base_path + 'gg100k_data.memmap'

    num_data = 100_000
    num_dims = 7_682

    data_memmap = np.memmap(
        filename=data_file,
        dtype=np.int8,
        mode='r',
        shape=(num_data, num_dims),
    )

    search_stats_file = 'manifold_logs/gg100k_search_stats.csv'
    if not os.path.exists(search_stats_file):
        with open(search_stats_file, 'w') as of:
            of.write('argquery, query_radius, current_depth, cluster_names\n')

    np.random.seed(42)
    manifold = Manifold(data_memmap, 'hamming', propagate=False, calculate_neighbors=False)
    old_depth = 50
    if old_depth > 0:
        s = time()
        with open(f'manifold_logs/chess_gg100k_{old_depth}.pickle', 'rb') as infile:
            manifold = manifold.pickle_load(fp=infile, data=data_memmap)
            manifold.__dict__['propagate'] = False
            manifold.__dict__['calculate_neighbors'] = False
            manifold.__dict__['search_stats_file'] = search_stats_file
        print(f'reading from json with depth {old_depth} took {time() - s:.4f} seconds.')

    max_search_depth = 35
    radii = [c.radius for c in manifold.graphs[max_search_depth]]
    query_radius = float(np.median(radii))
    for _ in range(10):
        argquery = int(np.random.choice(data_memmap.shape[0]))
        manifold.__dict__['argquery'] = argquery
        s = time()
        results = manifold.find_points(data_memmap[argquery], radius=query_radius)
        print(f'search for {argquery} at radius {query_radius} took {time() - s:.4f} seconds and produced {len(results)} results.')
    return


# noinspection DuplicatedCode
def benchmark_search_apo100k():
    base_path = '/scratch/nishaq/APOGEE2/'
    data_file = base_path + 'apo100k_data.memmap'

    num_data = 100_000
    num_dims = 8_575

    data_memmap = np.memmap(
        filename=data_file,
        dtype=np.float32,
        mode='r',
        shape=(num_data, num_dims),
    )

    search_stats_file = 'manifold_logs/apo100k_search_stats.csv'
    if not os.path.exists(search_stats_file):
        with open(search_stats_file, 'w') as of:
            of.write('argquery, query_radius, current_depth, cluster_names\n')

    np.random.seed(42)
    manifold = Manifold(data_memmap, 'euclidean', propagate=False, calculate_neighbors=False)
    old_depth = 50
    if old_depth > 0:
        s = time()
        with open(f'manifold_logs/chess_apogee2_100k_{old_depth}.pickle', 'rb') as infile:
            manifold = manifold.pickle_load(fp=infile, data=data_memmap)
            manifold.__dict__['propagate'] = False
            manifold.__dict__['calculate_neighbors'] = False
            manifold.__dict__['search_stats_file'] = search_stats_file
        print(f'reading from json with depth {old_depth} took {time() - s:.4f} seconds.')

    max_search_depth = 35
    radii = [c.radius for c in manifold.graphs[max_search_depth]]
    query_radius = float(np.median(radii))
    query_radius = min(2000., query_radius)
    for _ in range(10):
        argquery = int(np.random.choice(data_memmap.shape[0]))
        manifold.__dict__['argquery'] = argquery
        s = time()
        results = manifold.find_points(data_memmap[argquery], radius=query_radius)
        print(f'search for {argquery} at radius {query_radius} took {time() - s:.4f} seconds and produced {len(results)} results.')
    return


if __name__ == '__main__':
    print(f'Ready for benchmarks!')
    # benchmark_apogee2_100k()
    # benchmark_greengenes_100k()
    # benchmark_search_gg100k()
    # benchmark_search_apo100k()
