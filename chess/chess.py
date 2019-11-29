""" CHESS API

This class wraps the underlying Cluster structure with a convenient API.
"""
import json
from collections import Counter
from functools import lru_cache
from typing import Callable, List, Dict, Union, Set

import numpy as np

from chess import defaults
from .cluster import Cluster
from .graph import graph, subgraphs, connected_clusters
from .query import Query
from .search import search


class CHESS:
    """ Clustered Hierarchical Entropy-Scaling Search Object.
    """

    def __init__(
            self,
            data: Union[np.memmap, np.ndarray],
            metric: str,
            max_depth: int = defaults.MAX_DEPTH,
            min_points: int = defaults.MIN_POINTS,
            min_radius: defaults.RADII_DTYPE = defaults.MIN_RADIUS,
            stopping_criteria: Callable[[any], bool] = None,
            labels: List = None,
            root: Cluster = None
    ):
        self.data = data
        self.metric = metric
        self.max_depth = max_depth
        self.min_points = min_points
        self.min_radius = min_radius
        self.stopping_criteria = stopping_criteria

        # Classification data
        self.labels = labels
        frequencies = dict(Counter(self.labels))
        self.weights = {k: frequencies[k] / sum(frequencies.values()) for k in frequencies.keys()}

        self.root = root or Cluster(self.data, self.metric)

    def __str__(self):
        """
        :return: CSV-style string with two columns, name and an array of points, one row per leaf cluster.
        """
        return '\n'.join([
            'name, points',
            *[str(c) for c in self.root.leaves()]
        ])

    def __repr__(self):
        """
        :return: CSV-style string with more attributes, one row per cluster, generated inorder.
        """
        return '\n'.join([
            '\t'.join(['name', 'radius', 'argcenter', 'points']),
            *[repr(c) for c in self.root.inorder()]
        ])

    def __eq__(self, other):
        return self.metric == other.metric and self.root == other.root

    def __hash__(self):
        return hash(repr(self))

    def build(self, stopping_criteria: Callable[[any], bool] = None):
        """ Clusters points recursively until stopping_criteria returns True.

        :param stopping_criteria: callable function that takes a cluster and has additional user-defined stopping criteria.
        """
        self.root.make_tree(
            max_depth=self.max_depth,
            min_points=self.min_points,
            min_radius=self.min_radius,
            stopping_criteria=stopping_criteria or self.stopping_criteria
        )
        return

    def deepen(self, levels: int = 1):
        """ Adds upto num_levels of depth to cluster-tree. """
        max_depth = max(l.depth for l in self.root.leaves()) + levels
        [l.make_tree(
            max_depth=max_depth,
            min_points=self.min_points,
            min_radius=self.min_radius,
            stopping_criteria=self.stopping_criteria)
            for l in self.root.leaves()]
        return

    def search(self, query: np.ndarray, radius: defaults.RADII_DTYPE):
        """ Searches the clusters for all points within radius of query.
        """
        return search(self.root, Query(point=query, radius=radius))

    def select(self, cluster_name: str) -> Union[Cluster, None]:
        """ Returns the cluster with the given name.

        This function steps down the tree from the root node,
        until it either finds the requested node or runs out
        of nodes to check. At which point it returns either
        the node in the former case or None in the latter.

        :param cluster_name: the name of the requested cluster.
        :returns either a Cluster if it is found or None.
        """
        cluster_name = list(reversed(cluster_name))
        c = self.root
        while c and cluster_name:
            # Go either left or right.
            if cluster_name[-1] == '0':
                c = c.left
            elif cluster_name[-1] == '1':
                c = c.right
            else:
                raise ValueError(f'Invalid character in cluster name: {cluster_name[-1]}')

            # Then, advance to the next direction.
            cluster_name.pop()
        return c

    @lru_cache()
    def graph(self, depth: int = None) -> Dict[Cluster, Set[Cluster]]:
        """ Returns the graph representation of all leaf clusters at depth.

        :param depth: max-depth of the leaf-clusters in the graph.
        :return: dict of {cluster: neighbors} representing a graph.
        """
        return graph(list(self.root.leaves(depth)))

    @lru_cache()
    def subgraphs(self, depth: int = None) -> List[Dict[Cluster, Set[Cluster]]]:
        """ Returns the connected subgraphs of the given graph. """
        return subgraphs(self.graph(depth))

    @lru_cache()
    def subgraph(self, cluster: Cluster) -> Dict[Cluster, Set[Cluster]]:
        """ Returns the sub-graph to which a given cluster belongs. """
        depth: int = len(cluster.name)
        cc: List[Dict[Cluster, Set[Cluster]]] = self.subgraphs(depth)
        return next(filter(lambda c: cluster in set(c.keys()), cc))

    def connected_clusters(self, depth: int = None):
        """ Returns a list of sets of clusters, where each set contains all clusters from a component.
        """
        g = self.graph(depth)
        return connected_clusters(g)

    def compress(self, filename: str):
        """ Compresses the clusters.
        """
        mm = np.memmap(
            filename,
            dtype=self.root.data.dtype,
            mode='w+',
            shape=self.root.data.shape,
        )
        i = 0
        for cluster in self.root.leaves():
            points = cluster.compress()
            mm[i:i + len(points)] = points[:]
            i += len(points)
        mm.flush()
        return

    def write(self, filename: str):
        """ Writes the CHESS object to the given filename.
        """
        import inspect
        with open(filename, 'w') as f:
            json.dump(
                {
                    'metric': self.metric,
                    'max_depth': self.max_depth,
                    'min_points': self.min_points,
                    'min_radius': self.min_radius,
                    'stopping_criteria': str(inspect.getsource(self.stopping_criteria)) if self.stopping_criteria else '',
                    'labels': self.labels,
                    'root': self.root.json(),
                },
                f,
                indent=4,
            )
        return

    @staticmethod
    def load(filename: str, data: Union[np.memmap, np.ndarray], labels: List = None):
        """ Loads the CHESS object from the given file.
        """
        with open(filename, 'r') as f:
            d = json.load(f)
        # TODO: stopping criteria
        return CHESS(
            data=data,
            metric=d['metric'],
            max_depth=d['max_depth'],
            min_points=d['min_points'],
            min_radius=d['min_radius'],
            stopping_criteria=d['stopping_criteria'],
            labels=d['labels'],
            root=Cluster.from_json(d['root'], data),
        )

    def label_cluster_tree(self):
        """ Classifies each cluster in the cluster tree. """
        return {c: c.class_distribution(data_labels=self.labels, data_weights=self.weights) for c in self.root.inorder()}

    def label_cluster(self, cluster: Cluster) -> Dict:
        """ Returns the probability of the cluster having each label. """
        return cluster.class_distribution(data_labels=self.labels, data_weights=self.weights)
