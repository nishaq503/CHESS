from typing import Union, List, Dict, Set

import numpy as np

from chess import defaults
from chess.cluster import Cluster
from chess.distance import calculate_distances


class Graph:
    """ Defines the Graph class to investigate leaf-level properties of the data in CHESS. """

    def __init__(
            self,
            data: Union[np.memmap, np.ndarray],
            metric: str,
            leaves: List[Cluster],
    ):
        """ Initializes graph object on leaves.

        :param data: numpy memmap of points in leaves.
        :param metric: distance function used for clustering.
        :param leaves: list of leaf clusters with which to build the graph.
        """

        self.data: np.memmap = data
        assert len(data.shape) > 1, 'Got 1-d array as data. Expected at least 2-dim array.'
        assert data.shape[0] > 1, 'Got only one point in data. Expected more points.'

        self.metric: str = metric

        self.leaves: List[Cluster] = leaves
        assert len(leaves) > 1, 'Got only one leaf. Need more to build a graph.'

        self.graph: Dict[str, Set[str]] = {l.name: set() for l in leaves}

    def build(self):
        """ Builds the graph by adding edges between leaves that have overlap. """
        centers = np.asarray([l.center() for l in self.leaves], dtype=defaults.RADII_DTYPE)
        distances = calculate_distances(centers, centers, self.metric)
        radii = np.asarray([l.radius() for l in self.leaves], dtype=defaults.RADII_DTYPE)
        radii_matrix = (np.zeros_like(distances, dtype=defaults.RADII_DTYPE) + radii).T + radii
        edges = np.sign(distances - radii_matrix)
        np.fill_diagonal(edges, 0)
        left, right = tuple(map(list, np.where(edges < 0)))
        [self.graph[self.leaves[l].name].add(self.leaves[r].name) for l, r in zip(left, right)]
        return

    def connected_components(self) -> List[Set[Cluster]]:
        """ Finds the connected components in the graph.

        :return: List of sets of leaf names where each set is a connected component
        """
        def dft(start):
            visited_, stack = set(), [start]
            while stack:
                vertex = stack.pop()
                if vertex not in visited_:
                    visited_.add(vertex)
                    if vertex in self.graph.keys():
                        stack.extend(self.graph[vertex] - visited_)
            return visited_

        unvisited: Set[str] = set(self.graph.keys())
        components: List[Set[Cluster]] = []

        while unvisited:
            visited = dft(unvisited.pop())
            components.append(visited.copy())
            unvisited -= visited

        return components

    # TODO: Think about how to use number of components into a good stopping condition.
    # TODO: Implement max-flow and min-cut.
    # TODO: Implement methods to find bottlenecks and "almost disconnected components"
