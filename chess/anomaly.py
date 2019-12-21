from collections import deque
from typing import Dict, Set

import numpy as np

from chess import criterion
from chess.datasets import bullseye
from chess.manifold import Manifold, Graph, Cluster
from chess.types import Data


def outrank_anomalies(data: Data) -> Dict[int, float]:
    """ Determines anomalies by the Outrank algorithm.

    :param data: data in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    manifold: Manifold = Manifold(data, 'euclidean').build(criterion.MaxDepth(10))
    graph: Graph = manifold.graphs[-1]

    subgraphs: Set[Graph] = graph.subgraphs
    anomalies: Dict[int, float] = dict()
    for subgraph in subgraphs:
        results: Dict[Cluster, int] = subgraph.random_walk(steps=len(subgraph.clusters.keys()), walks=len(subgraph.clusters.keys()))
        num_steps = sum(results.values())
        results = {k: v / num_steps for k, v in results.items()}
        anomalies.update({p: v for c, v in results.items() for p in c.argpoints})

    return anomalies


def k_neighborhood_anomalies(data: Data, k: int) -> Dict[int, float]:
    """ Determines anomalies by the considering the graph-neighborhood of clusters.

    :param data: data in which to find anomalies.
    :param k: size of neighborhood to consider.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    manifold: Manifold = Manifold(data, 'euclidean').build(criterion.MaxDepth(10))
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


def cardinality_anomalies(data: Data) -> Dict[int, float]:
    """ Determines anomalies by the considering the cardinality of clusters in the graph.

    :param data: data in which to find anomalies.
    :return: Dictionary of indexes in the data with the confidence (in the range 0. to 1.) that the point is an anomaly.
    """
    manifold: Manifold = Manifold(data, 'euclidean').build(criterion.MaxDepth(10))
    graph: Graph = manifold.graphs[-1]

    min_points: int = int(np.percentile((len(c.argpoints) for c in graph.clusters), 0.02))
    results = {c: min_points / len(c.argpoints) for c in graph.clusters}
    anomalies: Dict[int, float] = {p: v for c, v in results.items() for p in c.argpoints}
    return anomalies


def main():
    data, _ = bullseye(n=1000, num_rings=3)
    labels = [0. for _ in range(data.shape[0])]

    noise: np.ndarray = np.random.rand(data.shape[0] // 50, data.shape[1])
    labels.extend([1. for _ in range(noise.shape[0])])

    data = np.concatenate([data, noise], axis=0)

    anomalies = outrank_anomalies(data)
    anomalies = sorted([(p, c) for p, c in anomalies.items()])
    anomalies = [c for _, c in anomalies]

    differences = [abs(l - a) for l, a in zip(labels, anomalies) if a > 0.5]
    successes = [d < 0.5 for d in differences]

    print(f'{len(successes) / len(differences)} of the noisy points were correctly identified as anomalies.')


if __name__ == '__main__':
    main()
