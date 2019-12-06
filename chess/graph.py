""" Graph Functions.

TODO: Think about how to use number of components into a good stopping condition.
TODO: Implement max-flow and min-cut.
TODO: Implement methods to find bottlenecks and "almost disconnected components"
TODO: Implement shortest-path methods as possible alternative to min-cut for finding bottleneck edges.
"""
from typing import List, Dict, Set

import numpy as np

from chess import defaults
from cluster import Cluster
from distance import calculate_distances
from query import Query


def chess_graph(chess_object, depth: int) -> Dict[Cluster, Set[Cluster]]:
    """
    Builds and returns a graph from the given list of clusters.
    Uses the CHESS-tree to gain an algorithmic improvement over naive construction.
    """
    return {c: set(tree_search(chess_object.root, Query(point=c.center(), radius=c.radius()), depth)) - {c}
            for c in chess_object.root.leaves(depth)}


def tree_search(cluster: Cluster, query: Query, depth) -> List[Cluster]:
    results = []
    if cluster.depth < depth:
        if cluster.left or cluster.right:
            results.extend(tree_search(cluster.left, query, depth))
            results.extend(tree_search(cluster.right, query, depth))
        elif query in cluster:
            results.append(cluster)
    elif query in cluster:
        results.append(cluster)
    return results


def graph(clusters: List[Cluster]) -> Dict[Cluster, Set[Cluster]]:
    """ Builds and returns a graph from the given list of clusters. """
    g: Dict[Cluster, Set[Cluster]] = {l: set() for l in clusters}
    centers = np.asarray([l.center() for l in clusters], dtype=defaults.RADII_DTYPE)
    distances = calculate_distances(centers, centers, clusters[0].metric)
    radii = np.asarray([l.radius() for l in clusters], dtype=defaults.RADII_DTYPE)
    radii_matrix = (np.zeros_like(distances, dtype=defaults.RADII_DTYPE) + radii).T + radii
    np.fill_diagonal(radii_matrix, 0)
    left, right = tuple(map(list, np.where(distances - radii_matrix <= 0)))
    [(g[clusters[l]].add(clusters[r]), )  # g[clusters[r]].add(clusters[l])
     for l, r in zip(left, right) if clusters[l].name != clusters[r].name]
    return g


def depth_first_traversal(g: Dict[Cluster, Set[Cluster]], start: Cluster) -> Set[Cluster]:
    """ Depth first traversal. """
    visited: Set[Cluster] = set()
    stack: List[Cluster] = [start]
    while stack:
        vtx: Cluster = stack.pop()
        if vtx not in visited:
            visited.add(vtx)
            stack.extend(g[vtx] - visited)
    return visited


def connected_clusters(g: Dict[Cluster, Set[Cluster]]) -> List[Set[Cluster]]:
    """ Returns a list of sets where each set contains all clusters in a connected component. """
    unvisited: Set[Cluster] = set(g.keys())
    components: List[Set[Cluster]] = []

    while unvisited:
        visited = depth_first_traversal(g, unvisited.pop())
        components.append(visited)
        unvisited -= visited

    return components


def subgraphs(g: Dict[Cluster, Set[Cluster]]) -> List[Dict[Cluster, Set[Cluster]]]:
    """ Returns all connected subgraphs from the given graph. """
    components = connected_clusters(g)
    return [{cluster: g[cluster] for cluster in component} for component in components]
