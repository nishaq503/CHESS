from time import time
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

from chess.manifold import Cluster, Manifold
import benchmarks


def get_embedding(dataset: str) -> np.ndarray:
    if dataset == 'apogee':
        filename = '/scratch/nishaq/APOGEE2/apo100k_umap.memmap'
        dtype = np.float32
    elif dataset == 'greengenes':
        filename = '/scratch/nishaq/GreenGenes/gg100k_umap.memmap'
        dtype = np.float32
    else:
        raise ValueError(f'dataset must be apogee or greengenes. Got {dataset} instead.')

    embedding_memmap: np.memmap = np.memmap(filename=filename, dtype=dtype, mode='r', shape=(100_000, 3))
    embedding: np.ndarray = np.zeros_like(embedding_memmap, dtype=dtype)
    embedding[:] = embedding_memmap[:]
    return embedding


def get_data(dataset: str) -> np.memmap:
    if dataset == 'apogee':
        filename = '/scratch/nishaq/APOGEE2/apo100k_data.memmap'
        dtype = np.float32
        shape = (100_000, 8_575)
    elif dataset == 'greengenes':
        filename = '/scratch/nishaq/GreenGenes/gg100k_data.memmap'
        dtype = np.int8
        shape = (100_000, 7_682)
    else:
        raise ValueError(f'dataset must be apogee or greengenes. Got {dataset} instead.')

    return np.memmap(
        filename=filename,
        dtype=dtype,
        mode='r',
        shape=shape
    )


def plot(angles, data, limits, labels, folder, figsize, dpi, s, title):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=labels, s=s, cmap='Set1')
    plt.title(title)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_zlim(limits[2])
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)
    for azimuth in range(angles[0], angles[1]):
        ax.view_init(elev=10, azim=azimuth)
        plt.savefig(folder + f'{benchmarks.PLOT_NUMBER:05d}.png', bbox_inches='tight', pad_inches=0)
        benchmarks.PLOT_NUMBER += 1
        ax.view_init(elev=10, azim=azimuth + 0.5)
        plt.savefig(folder + f'{benchmarks.PLOT_NUMBER:05d}.png', bbox_inches='tight', pad_inches=0)
        benchmarks.PLOT_NUMBER += 1
        # break  # TODO: remove
    plt.close('all')
    return


def get_labels() -> Dict[int, int]:
    labels = {i: 8 for i in range(100_000)}
    labels[0] = 0  # red, query
    labels[1] = 1  # blue
    labels[2] = 2  # green, potential points
    labels[3] = 3  # purple
    labels[4] = 4  # orange
    labels[5] = 5  # yellow
    labels[6] = 6  # brown
    labels[7] = 7  # pink
    labels[8] = 8  # gray, background
    return labels


# noinspection DuplicatedCode
def query_plots(
        umap_data: np.ndarray,
        argquery: int,
        depth_to_clusters: Dict[int, Tuple[List[Cluster], List[Cluster]]],
        base_path: str,
):
    limits = (
        [int(np.min(umap_data[:, 0])), int(np.max(umap_data[:, 0]))],
        [int(np.min(umap_data[:, 1])), int(np.max(umap_data[:, 1]))],
        [int(np.min(umap_data[:, 2])), int(np.max(umap_data[:, 2]))],
    )
    angle, step = 0, 30
    depths = [0] * 6
    depths.extend(range(0, 51))
    fractions = {}
    for d in depths:
        if depth_to_clusters:
            if d in depth_to_clusters:
                gray, green = depth_to_clusters[d]
                gray_argpoints: List[int] = [int(p) for cluster in gray for p in cluster.argpoints]
                gray_points = umap_data[gray_argpoints]
                gray_labels = [8 if g != argquery else 0 for g in gray_argpoints]
                fractions[d] = len(gray_argpoints) / 100_000
                print(f'depth: {d}, fraction: {fractions[d]:.6f}')
                if len(gray_labels) >= 9:
                    for g in range(9):
                        if gray_argpoints[g] == argquery:
                            gray_labels[9] = g
                        else:
                            gray_labels[g] = g
                title = '100,000 points from the APOGEE data under UMAP dimensionality reduction'
                if d != 0:
                    title = f'depth: {d}, fraction: {fractions[d]:.6f}'
                plot(
                    angles=(angle, angle + step),
                    data=gray_points,
                    limits=limits,
                    labels=gray_labels,
                    folder=base_path,
                    figsize=(12, 12),
                    dpi=150,
                    s=2. if fractions[d] < 0.1 else 0.1,
                    title=title
                )
                angle += step
                if d != 0:
                    green_argpoints: List[int] = [int(p) for cluster in green for p in cluster.argpoints]
                    green_set = set(green_argpoints)
                    gray_labels = [2 if g in green_set else 8 for g in gray_argpoints]
                if len(gray_labels) >= 9:
                    for g in range(9):
                        if gray_argpoints[g] == argquery:
                            gray_labels[9] = g
                        else:
                            gray_labels[g] = g
                plot(
                    angles=(angle, angle + step),
                    data=gray_points,
                    limits=limits,
                    labels=gray_labels,
                    folder=base_path,
                    figsize=(12, 12),
                    dpi=150,
                    s=2. if fractions[d] < 0.1 else 0.1,
                    title=title
                )
                angle += step
        else:
            break
    # print(f'argquery: {argquery}, fractions: {sorted(list(fractions.items()))}')
    return


def make_plots(
        dataset: str,
        manifold: Manifold,
        meta_data: Dict[int, Dict[int, Tuple[List[str], List[str]]]]
):
    if dataset == 'apogee':
        embedding = get_embedding(dataset)
        plot_folder = '../presentation/apogee2/umap/'
    elif dataset == 'greengenes':
        embedding = get_embedding(dataset)
        plot_folder = '../presentation/greengenes/tsne/'
    else:
        raise ValueError(f'dataset must be either apogee or greengenes. Got {dataset} instead.')
    for q in meta_data.keys():
        if q == 6265:
            depth_to_cluster_names: Dict[int, Tuple[List[str], List[str]]] = meta_data[q]
            depth_to_clusters: Dict[int, Tuple[List[Cluster], List[Cluster]]] = dict()
            for k, v in depth_to_cluster_names.items():
                try:
                    grays = [manifold.select(c) for c in v[0]]
                    greens = [manifold.select(c) for c in v[1]]
                except AssertionError:
                    continue
                depth_to_clusters[k] = (grays, greens)
            else:
                query_plots(
                    umap_data=embedding,
                    argquery=q,
                    depth_to_clusters=depth_to_clusters,
                    base_path=plot_folder
                )
            # break  # TODO: remove
    return


def get_metadata(dataset: str) -> Dict[int, Dict[int, Tuple[List[str], List[str]]]]:
    if dataset == 'apogee':
        search_file = 'manifold_logs/apo100k_search_stats.csv'
        search_stats: pd.DataFrame = pd.read_csv(search_file, sep=', ')
    elif dataset == 'greengenes':
        search_file = 'manifold_logs/gg100k_search_stats.csv'
        search_stats: pd.DataFrame = pd.read_csv(search_file, sep=', ')
    else:
        raise ValueError(f'dataset must be either apogee or greengenes. Got {dataset} instead.')

    argqueries = sorted(list(set(search_stats.argquery.values)))

    df_by_argquery: Dict[int, Dict[int, Tuple[List[str], List[str]]]] = dict()
    for aq in argqueries:
        temp_df: pd.DataFrame = search_stats[search_stats.argquery == aq]
        temp_df.reset_index(drop=True, inplace=True)
        temp_df.drop(labels='argquery', axis=1, inplace=True)
        temp_df.drop(labels='query_radius', axis=1, inplace=True)

        df_by_argquery[aq] = dict()
        for d in range(0, 51):
            d_df: pd.DataFrame = temp_df[temp_df.current_depth == d]
            d_df.reset_index(drop=True, inplace=True)
            names: List[List[str]] = [list(), list()]
            for j, row in d_df.iterrows():
                cluster_names = row.cluster_names
                names[j] = list(cluster_names.split(';'))
            df_by_argquery[aq][d] = (names[0], names[1])

    return df_by_argquery


def get_manifold(dataset: str) -> Manifold:
    data = get_data(dataset)
    s = time()
    if dataset == 'apogee':
        with open('manifold_logs/chess_apogee2_100k_50.pickle', 'rb') as infile:
            manifold: Manifold = Manifold.pickle_load(infile, data)
    elif dataset == 'greengenes':
        with open('manifold_logs/chess_gg100k_50.pickle', 'rb') as infile:
            manifold: Manifold = Manifold.pickle_load(infile, data)
    print(f'reading {dataset} manifold from pickle took {time() - s:.6f} seconds.')

    manifold.__dict__['propagate'] = False
    manifold.__dict__['calculate_neighbors'] = False
    return manifold


def main():
    dataset = 'apogee'
    manifold: Manifold = get_manifold(dataset=dataset)
    meta_data = get_metadata(dataset=dataset)
    make_plots(dataset=dataset, manifold=manifold, meta_data=meta_data)


if __name__ == '__main__':
    main()
