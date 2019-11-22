""" Graph Functions.

TODO: Think about how to use number of components into a good stopping condition.
TODO: Implement max-flow and min-cut.
TODO: Implement methods to find bottlenecks and "almost disconnected components"
"""
from typing import List, Dict, Set

import numpy as np

from chess import defaults
from chess.cluster import Cluster
from chess.distance import calculate_distances


def graph(clusters: List[Cluster]) -> Dict:
    """ Builds and returns a graph from the given list of clusters. """
    g = {l: set() for l in clusters}
    centers = np.asarray([l.center() for l in clusters], dtype=defaults.RADII_DTYPE)
    distances = calculate_distances(centers, centers, clusters[0].metric)
    radii = np.asarray([l.radius() for l in clusters], dtype=defaults.RADII_DTYPE)
    radii_matrix = (np.zeros_like(distances, dtype=defaults.RADII_DTYPE) + radii).T + radii
    np.fill_diagonal(radii_matrix, 0)
    left, right = tuple(map(list, np.where((distances - radii_matrix) < 0)))
    # TODO: Should we store clusters here, or just name?
    [g[clusters[l]].add(clusters[r]) for l, r in zip(left, right)]
    return g


def depth_first_traversal(graph, start) -> Set[Cluster]:
    """ Depth first traversal. """
    visited, stack = set(), [start]
    while stack:
        vtx = stack.pop()
        if vtx not in visited:
            visited.add(vtx)
            stack.extend(graph[vtx] - visited)
    return visited


def connected_components(graph) -> List[Set[Cluster]]:
    """ Returns all connected components from the given list of clusters. """

    unvisited: Set[str] = set(graph.keys())
    components: List[Set[Cluster]] = []

    while unvisited:
        visited = depth_first_traversal(graph, unvisited.pop())
        components.append(visited.copy())
        unvisited -= visited

    return components
