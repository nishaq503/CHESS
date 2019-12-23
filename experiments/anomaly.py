from collections import deque
from typing import Dict, Set, List

import matplotlib.pyplot as plt
import numpy as np

from chess import criterion
from chess.datasets import bullseye
from chess.manifold import Manifold, Graph, Cluster


def min_max_normalize(measures: Dict[int, float]) -> Dict[int, float]:
    min_v, max_v = min(measures.values()), max(measures.values())
    return {k: (v - min_v) / (max_v - min_v) for k, v in measures.items()}


def mean_normalize(measures: Dict[int, float]) -> Dict[int, float]:
    min_v, max_v, mean_v = min(measures.values()), max(measures.values()), np.mean(list(measures.values()))
    return {k: (v - mean_v) / (max_v - min_v) for k, v in measures.items()}


def z_score_normalization(measures: Dict[int, float]) -> Dict[int, float]:
    mean_v, std_v = np.mean(list(measures.values())), np.std(list(measures.values()))
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


def outrank_anomalies(manifold: Manifold, normalization: str) -> Dict[int, float]:
    """ Determines anomalies by the Outrank algorithm.

    :param manifold: manifold in which to find anomalies.
    :param normalization: type of normalization to use on the measures of anomalousness.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    graph: Graph = manifold.graphs[-1]

    subgraphs: Set[Graph] = graph.subgraphs
    anomalies: Dict[int, float] = dict()
    for subgraph in subgraphs:
        results: Dict[Cluster, int] = subgraph.random_walk(steps=len(subgraph.clusters.keys()), walks=len(subgraph.clusters.keys()))
        anomalies.update({p: v for c, v in results.items() for p in c.argpoints})

    anomalies = normalize(anomalies, normalization)
    return {k: 1. - v for k, v in anomalies.items()}


def k_neighborhood_anomalies(manifold: Manifold, k: int) -> Dict[int, float]:
    """ Determines anomalies by the considering the graph-neighborhood of clusters.

    :param manifold: manifold in which to find anomalies.
    :param k: size of neighborhood to consider.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    graph: Graph = manifold.graphs[-1]

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
    max_neighborhood = max(results.values())
    anomalies: Dict[int, float] = {p: (1. - v / max_neighborhood) for c, v in results.items() for p in c.argpoints}
    return anomalies


def cardinality_anomalies(manifold: Manifold, normalization: str) -> Dict[int, float]:
    """ Determines anomalies by the considering the cardinality of clusters in the graph.

    :param manifold: Manifold in which to find anomalies.
    :param normalization: which method to use for normalization.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    graph = manifold.graphs[-1]
    results = {c: len(c.argpoints) for c in graph.clusters.keys()}
    anomalies: Dict[int, float] = {p: v for c, v in results.items() for p in c.argpoints}
    anomalies = normalize(anomalies, normalization)
    return {p: 1. - v for p, v in anomalies.items()}


def plot_histogram(x: List[float], method: str, depth: int, mode: str):
    plt.clf()
    fig = plt.figure()
    n, bins, patches = plt.hist(x=x, bins=20, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Anomalousness')
    plt.ylabel('Counts')
    plt.title(f'method: {method}, depth: {depth}, mode: {mode}.')
    max_freq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(max_freq / 10) * 10 if max_freq % 10 else max_freq + 10)
    # plt.show()
    fig.savefig(f'../plots/anomaly/{method}_{depth}_{mode}.png')
    return


def main():
    data, _ = bullseye(n=1000, num_rings=3)

    num_noise_points = data.shape[0] // 50
    noise: np.ndarray = np.random.rand(num_noise_points, data.shape[1])

    noisy_data = np.concatenate([data, noise], axis=0)

    manifold: Manifold = Manifold(noisy_data, 'euclidean')
    manifold.build(
        criterion.MinRadius(0.05),
        criterion.MaxDepth(6)
    )

    for depth in range(6, 15):
        manifold.build_tree(criterion.MinRadius(0.1), criterion.MaxDepth(depth))
        manifold.build_graph(depth)

        for normalization in ['min-max', 'mean', 'z-score', None]:
            anomalies = cardinality_anomalies(manifold, normalization)
            plot_histogram(
                x=[v for _, v in anomalies.items()],
                method='cardinality',
                depth=depth,
                mode=normalization,
            )

    # anomalies = sorted([(v, p) for p, v in anomalies.items()])
    # anomalies = [(0, p) if i < data.shape[0] else (1, p) for i, (p, _) in enumerate(anomalies)]
    # anomalies = sorted([(p, v) for v, p in anomalies])

    # true_labels = [0 if i < data.shape[0] else 1 for i in range(noisy_data.shape[0])]
    # pred_labels = [v for _, v in anomalies]

    # matrix = confusion_matrix(true_labels, pred_labels)
    # print(matrix)

    return


if __name__ == '__main__':
    main()
