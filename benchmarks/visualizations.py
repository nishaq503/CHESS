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
        dtype = np.int8
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


def plot(angles, data_, labels, folder, figsize, dpi, s, title):
    x, y, z = data_[:, 0], data_[:, 1], data_[:, 2]
    x_limits = [int(np.min(x)), int(np.max(x))]
    y_limits = [int(np.min(y)), int(np.max(y))]
    z_limits = [int(np.min(z)), int(np.max(z))]
    plt.clf()
    plt.close('all')
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=labels, s=s, cmap='Set1')
    plt.title(title)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)
    for azimuth in range(angles[0], angles[1]):
        ax.view_init(elev=10, azim=azimuth)
        plt.savefig(folder + f'{benchmarks.PLOT_NUMBER:05d}.png', bbox_inches='tight', pad_inches=0)
        benchmarks.PLOT_NUMBER += 1
        break
    plt.close('all')
    return


def get_labels() -> Dict[int, int]:
    labels = {i_: 8 for i_ in range(100_000)}
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
    angle, step = 0, 15
    for d_ in range(0, 41):
        if d_ < 15:
            continue
        if depth_to_clusters:
            gray, green = depth_to_clusters[d_]
            gray_argpoints: List[int] = [int(p) for cluster in gray for p in cluster.argpoints]
            gray_points = umap_data[gray_argpoints]
            gray_labels = [8 if g != argquery else 0 for g in gray_argpoints]
            if len(gray_labels) >= 9:
                for g in range(9):
                    if gray_argpoints[g] == argquery:
                        gray_labels[9] = g
                    else:
                        gray_labels[g] = g
            fraction: float = len(gray_argpoints) / 100_000
            plot(
                angles=(angle, angle + step),
                data_=gray_points,
                labels=gray_labels,
                folder=base_path,
                figsize=(12, 12),
                dpi=150,
                s=2. if fraction < 0.1 else 0.1,
                title=f'argquery: {argquery} depth: {d_}, fraction: {fraction:.6f}'
            )
            angle += step
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
                data_=gray_points,
                labels=gray_labels,
                folder=base_path,
                figsize=(12, 12),
                dpi=150,
                s=2. if fraction < 0.1 else 0.1,
                title=f'argquery: {argquery} depth: {d_}, fraction: {fraction:.6f}'
            )
            angle += step
        else:
            break
    return


def make_apogee_plots(
        manifold: Manifold,
        meta_data: Dict[int, Dict[int, Tuple[List[str], List[str]]]]
):
    embedding = get_embedding('apogee')
    plot_folder = '../presentation/apogee2/umap/'
    for i_, q in enumerate(meta_data.keys()):
        depth_to_cluster_names: Dict[int, Tuple[List[str], List[str]]] = meta_data[q]
        depth_to_clusters: Dict[int, Tuple[List[Cluster], List[Cluster]]] = dict()
        for k, v in depth_to_cluster_names.items():
            grays = [manifold.select(c) for c in v[0]]
            greens = [manifold.select(c) for c in v[1]]
            depth_to_clusters[k] = (grays, greens)
        query_plots(
            umap_data=embedding,
            argquery=q,
            depth_to_clusters=depth_to_clusters,
            base_path=plot_folder
        )
    return


def get_apogee_metadata() -> Dict[int, Dict[int, Tuple[List[str], List[str]]]]:
    search_file = 'manifold_logs/apo100k_search_stats.csv'
    search_stats: pd.DataFrame = pd.read_csv(search_file, sep=', ')

    argqueries = sorted(list(set(search_stats.argquery.values)))

    df_by_argquery: Dict[int, Dict[int, Tuple[List[str], List[str]]]] = dict()
    for i, aq in enumerate(argqueries):
        temp_df: pd.DataFrame = search_stats[search_stats.argquery == aq]
        temp_df.reset_index(drop=True, inplace=True)
        temp_df.drop(labels='argquery', axis=1, inplace=True)
        temp_df.drop(labels='query_radius', axis=1, inplace=True)

        df_by_argquery[aq] = dict()
        for d in range(0, 41):
            d_df: pd.DataFrame = temp_df[temp_df.current_depth == d]
            d_df.reset_index(drop=True, inplace=True)
            names: List[List[str]] = [list(), list()]
            for j, row in d_df.iterrows():
                cluster_names = row.cluster_names
                names[j] = list(cluster_names.split(';'))
            df_by_argquery[aq][d] = (names[0], names[1])

    return df_by_argquery


def get_apogee_manifold() -> Manifold:
    data = get_data('apogee')
    with open('manifold_logs/chess_apogee2_100k_50.pickle', 'rb') as infile:
        manifold: Manifold = Manifold.pickle_load(infile, data)

    manifold.__dict__['propagate'] = False
    manifold.__dict__['calculate_neighbors'] = False
    return manifold


if __name__ == '__main__':
    m: Manifold = get_apogee_manifold()
    searches_meta_data = get_apogee_metadata()
    make_apogee_plots(m, searches_meta_data)
