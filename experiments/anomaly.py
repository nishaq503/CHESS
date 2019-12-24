import os
import random
from collections import deque
from typing import Dict, Set, List

import matplotlib.pyplot as plt
import numpy as np
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

from chess import criterion
from chess.datasets import bullseye, spiral_2d, tori, skewer
from chess.manifold import Manifold, Graph, Cluster


def min_max_normalize(measures: Dict[int, float]) -> Dict[int, float]:
    min_v, max_v = np.min(list(measures.values())), np.max(list(measures.values()))
    min_v, max_v, = float(min_v), float(max_v)
    if min_v == max_v:
        max_v += 1.
    return {k: (v - min_v) / (max_v - min_v) for k, v in measures.items()}


def mean_normalize(measures: Dict[int, float]) -> Dict[int, float]:
    min_v, max_v, mean_v = np.min(list(measures.values())), np.max(list(measures.values())), np.mean(list(measures.values()))
    min_v, max_v, mean_v = float(min_v), float(max_v), float(mean_v)
    if min_v == max_v:
        max_v += 1.
    return {k: (v - mean_v) / (max_v - min_v) for k, v in measures.items()}


def z_score_normalization(measures: Dict[int, float]) -> Dict[int, float]:
    mean_v, std_v = np.mean(list(measures.values())), np.std(list(measures.values()))
    mean_v, std_v = float(mean_v), float(std_v)
    std_v = max(std_v, 1.)
    return {k: (v - mean_v) / std_v for k, v in measures.items()}


def normalize(anomalies: Dict[int, float], normalization: str) -> Dict[int, float]:
    if normalization == 'min-max':
        anomalies = min_max_normalize(anomalies)
    elif normalization == 'mean':
        anomalies = mean_normalize(anomalies)
    elif normalization == 'z-score':
        anomalies = z_score_normalization(anomalies)
    elif normalization is not None:
        raise ValueError(f'normalization mode must be one of "min-max", "mean", "z-score", or None. Got {normalization} instead.')
    return anomalies


def outrank_anomalies(
        graph: Graph,
        normalization: str,
) -> Dict[int, float]:
    """ Determines anomalies by the Outrank algorithm.

    :param graph: manifold in which to find anomalies.
    :param normalization: type of normalization to use on the measures of anomalousness.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    subgraphs: Set[Graph] = graph.subgraphs
    anomalies: Dict[int, float] = dict()
    for subgraph in subgraphs:
        results: Dict[Cluster, int] = subgraph.random_walk(
            steps=max(len(subgraph.clusters.keys()) // 10, 10),
            walks=max(len(subgraph.clusters.keys()) // 10, 10),
        )
        anomalies.update({p: v for c, v in results.items() for p in c.argpoints})

    anomalies = normalize(anomalies, normalization)
    return {k: 1. - v for k, v in anomalies.items()}


def k_neighborhood_anomalies(
        graph: Graph,
        normalization: str,
        k: int = 3,
) -> Dict[int, float]:
    """ Determines anomalies by the considering the graph-neighborhood of clusters.

    :param graph: manifold in which to find anomalies.
    :param normalization: which method to use for normalization.
    :param k: size of neighborhood to consider.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    def bft(start: Cluster) -> int:
        visited = set()
        queue = deque([start])
        for _ in range(k):
            if queue:
                c = queue.popleft()
                if c not in visited:
                    visited.add(c)
                    [queue.append(neighbor) for neighbor in c.neighbors.keys()]
            else:
                break
        return len(visited)

    results = {c: bft(c) for c in graph.clusters}
    anomalies: Dict[int, float] = {p: v for c, v in results.items() for p in c.argpoints}
    anomalies = normalize(anomalies, normalization)
    return {k: 1. - v for k, v in anomalies.items()}


def cluster_cardinality_anomalies(
        graph: Graph,
        normalization: str,
) -> Dict[int, float]:
    """ Determines anomalies by the considering the cardinality of clusters in the graph.

    :param graph: Manifold in which to find anomalies.
    :param normalization: which method to use for normalization.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    anomalies: Dict[int, float] = {
        p: len(c.argpoints)
        for c in graph.clusters.keys()
        for p in c.argpoints
    }
    anomalies = normalize(anomalies, normalization)
    return {p: 1. - v for p, v in anomalies.items()}


def subgraph_cardinality_anomalies(
        graph: Graph,
        normalization: str,
) -> Dict[int, float]:
    """ Determines anomalies by the considering the cardinality of connected components in the graph.

    :param graph: Manifold in which to find anomalies.
    :param normalization: which method to use for normalization.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    anomalies: Dict[int, float] = {
        p: len(subgraph.clusters.keys())
        for subgraph in graph.subgraphs
        for c in subgraph.clusters.keys()
        for p in c.argpoints
    }
    anomalies = normalize(anomalies, normalization)
    return {p: 1. - v for p, v in anomalies.items()}


def plot_histogram(
        x: List[float],
        dataset: str,
        method: str,
        depth: int,
        mode: str,
        save: bool,
):
    plt.clf()
    fig = plt.figure()
    n, bins, patches = plt.hist(x=x, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Anomalousness')
    plt.ylabel('Counts')
    plt.title(f'dataset: {dataset}, method: {method}, depth: {depth}, mode: {mode}')
    max_freq = n.max()
    plt.ylim(ymax=np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10)
    if save is True:
        filepath = f'../plots/anomaly/{dataset}/{method}/{mode}/{depth}_histogram.png'
        make_folders(dataset, method, mode)
        fig.savefig(filepath)
    else:
        plt.show()
    return


def plot_confusion_matrix(
        true_labels: List[int],
        anomalies: Dict[int, float],
        dataset: str,
        method: str,
        depth: int,
        mode: str,
        save: bool,
):
    p = float(sum(true_labels))
    n = float(len(true_labels) - p)

    tp_median = float(np.mean([v for k, v in anomalies.items() if true_labels[k] == 1]))
    tn_median = float(np.mean([v for k, v in anomalies.items() if true_labels[k] == 0]))
    tp = float(sum([1 if v > tp_median else 0 for k, v in anomalies.items() if true_labels[k] == 1]))
    tn = float(sum([1 if v <= tn_median else 0 for k, v in anomalies.items() if true_labels[k] == 0]))

    tpr, tnr = tp / p, tn / n
    fpr, fnr = 1 - tnr, 1 - tpr

    confusion_matrix = [[tpr, fpr], [fnr, tnr]]

    plt.clf()
    fig = plt.figure()
    # noinspection PyUnresolvedReferences
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Wistia)
    class_names = ['Normal', 'Anomaly']
    plt.title(f'dataset: {dataset}, method: {method}, depth: {depth}, mode: {mode}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    s = [['TPR', 'FPR'], ['FNR', 'TNR']]
    [plt.text(j, i, f'{s[i][j]} = {confusion_matrix[i][j]:.3f}', position=(-0.2 + j, 0.03 + i))
     for i in range(2)
     for j in range(2)]

    if save is True:
        filepath = f'../plots/anomaly/{dataset}/{method}/{mode}/{depth}_confusion_matrix.png'
        make_folders(dataset, method, mode)
        fig.savefig(filepath)
    else:
        plt.show()
    return


def make_folders(dataset, method, mode):
    dir_paths = [f'../plots',
                 f'../plots/anomaly',
                 f'../plots/anomaly/{dataset}',
                 f'../plots/anomaly/{dataset}/{method}',
                 f'../plots/anomaly/{dataset}/{method}/{mode}']
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    return


def plot_data(data: np.ndarray, labels: List[int], name: str):
    plt.clf()
    fig = plt.figure(figsize=(6, 6), dpi=300)
    title = name.split('/')[-1][:-4]
    if data.shape[1] == 2:
        ax = fig.add_subplot(111)
        plt.title(title)
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='Dark2', s=5.)
        ax.set_xlim([np.min(data[:, 0]), np.max(data[:, 0])])
        ax.set_ylim([np.min(data[:, 1]), np.max(data[:, 1])])
    elif data.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        plt.title(title)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=5., cmap='Dark2')
        ax.set_xlim([np.min(data[:, 0]), np.max(data[:, 0])])
        ax.set_ylim([np.min(data[:, 1]), np.max(data[:, 1])])
        ax.set_zlim([np.min(data[:, 2]), np.max(data[:, 2])])
        ax.view_init(elev=20, azim=60)

    plt.savefig(name, bbox_inches='tight', pad_inches=0.25)
    plt.show()
    plt.close('all')
    return


def main():
    datasets = {
        'bullseye': bullseye,
        'spiral': spiral_2d,
        'interlocking_tori': tori,
        'skewer': skewer,
    }
    methods = {
        'outrank': outrank_anomalies,
        'cluster_cardinality': cluster_cardinality_anomalies,
        'subgraph_cardinality': subgraph_cardinality_anomalies,
        'k_neighborhood': k_neighborhood_anomalies,
    }
    for dataset in datasets.keys():
        np.random.seed(42)
        random.seed(42)
        data, _ = datasets[dataset]()
        labels = [0 for _ in range(data.shape[0])]

        num_noise_points = (data.shape[0] // 250) * min((data.shape[1] ** 2), 10)
        # num_noise_points = num_noise_points // num_noise_points
        noise: np.ndarray = np.zeros(shape=(num_noise_points, data.shape[1]))
        for axis in range(noise.shape[1]):
            min_v, max_v = np.min(data[:, axis]), np.max(data[:, axis])
            noise[:, axis] = np.random.rand(num_noise_points, ) * (max_v - min_v) + min_v

        noisy_data = np.concatenate([data, noise], axis=0)
        labels.extend([1 for _ in range(num_noise_points)])

        plot_data(noisy_data, labels, f'../plots/anomaly/{dataset}.png')

        manifold: Manifold = Manifold(noisy_data, 'euclidean')
        if not os.path.exists(f'logs'):
            os.mkdir(f'logs')

        min_radius, max_depth, min_points = 0.1, 20, 1
        manifold.build(
            criterion.MinRadius(min_radius),
            criterion.MaxDepth(max_depth),
            criterion.MinPoints(min_points),
        )
        # filepath = f'logs/{dataset}_{noisy_data.shape[0]}_{min_radius}_{max_depth}_{min_points}.pickle'
        # if os.path.exists(filepath):
        #     with open(filepath, 'rb') as infile:
        #         manifold = manifold.load(infile, noisy_data)
        # else:
        #     manifold.build(
        #         criterion.MinRadius(min_radius),
        #         criterion.MaxDepth(max_depth),
        #         criterion.MinPoints(min_points),
        #     )
        #     with open(filepath, 'wb') as infile:
        #         manifold.dump(infile)

        print(f'\ndataset: {dataset}')
        for depth in range(6, manifold.depth + 1):
            print(f'depth: {depth},'
                  f' num_subgraphs: {len(manifold.graphs[depth].subgraphs)},'
                  f' num_clusters: {len(manifold.graphs[depth].clusters.keys())}')
            for method in methods.keys():
                for normalization in ['min-max', 'mean', 'z-score', None]:
                    anomalies = methods[method](manifold.graphs[depth], normalization)
                    plot_histogram(
                        x=[v for _, v in anomalies.items()],
                        dataset=dataset,
                        method=method,
                        depth=depth,
                        mode=normalization,
                        save=True,
                    )
                    plot_confusion_matrix(
                        true_labels=labels,
                        anomalies=anomalies,
                        dataset=dataset,
                        method=method,
                        depth=depth,
                        mode=normalization,
                        save=True,
                    )
                    plt.close('all')
    return


if __name__ == '__main__':
    main()
