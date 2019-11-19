""" CHESS API

This class wraps the underlying Cluster structure with a convenient API.
"""
import pickle
from typing import Callable, List

import numpy as np

from .cluster import Cluster
from .knn_search import knn_search
from .query import Query
from .search import search
from chess import defaults


def update_defaults(
        max_depth: int = None,
        min_points: int = None,
        min_radius: defaults.RADII_DTYPE = None,
        stopping_criteria: Callable[[any], bool] = None,
):
    # TODO: consider updating other defaults.
    # TODO: Consider moving this function elsewhere if it makes more sense.
    # TODO: Consider exposing this function in __init__.py.
    defaults.MAX_DEPTH = max_depth if max_depth is not None else defaults.MAX_DEPTH
    defaults.MIN_POINTS = min_points if min_points is not None else defaults.MIN_POINTS
    defaults.MIN_RADIUS = min_radius if min_radius is not None else defaults.MIN_RADIUS
    defaults.STOPPING_CRITERIA = stopping_criteria if stopping_criteria is not None else defaults.STOPPING_CRITERIA


class CHESS:
    """ Clustered Hierarchical Entropy-Scaling Search Object.
    """

    def __init__(
            self,
            data: np.memmap,
            metric: str,
            max_depth: int = None,
            min_points: int = None,
            min_radius: defaults.RADII_DTYPE = None,
            stopping_criteria: Callable[[any], bool] = None,
    ):
        update_defaults(max_depth, min_points, min_radius, stopping_criteria)

        self.data = data
        self.metric = metric

        self.root = Cluster(self.data, self.metric)

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
            'name, number_of_points, center, radius, lfd, is_leaf',
            *[repr(c) for c in self.root.inorder()]
        ])

    def __eq__(self, other):
        return self.metric == other.metric and self.root == other.root

    def build(
            self,
            max_depth: int = None,
            min_points: int = None,
            min_radius: defaults.RADII_DTYPE = None,
            stopping_criteria: Callable[[any], bool] = None,
    ):
        """ Clusters points recursively until stopping_criteria returns True.

        :param max_depth: max depth for the cluster-tree.
        :param min_points: minimum number of points in a cluster for it to be partitionable.
        :param min_radius: minimum radius of a cluster for it to be partitionable.
        :param stopping_criteria: callable function that takes a cluster and has additional user-defined stopping criteria.
        """
        # TODO: Should we update defaults here? Najib: leaning no.
        update_defaults(max_depth, min_points, min_radius, stopping_criteria)

        self.root = Cluster(self.data, self.metric)
        self.root.make_tree()
        return

    def add_levels(self, num_levels: int = 1):
        """ Adds upto num_levels of depth to cluster-tree. """
        leaves: List[Cluster] = list(self.root.leaves())
        old_depth = max(l.depth for l in leaves)

        max_depth = max(defaults.MAX_DEPTH, old_depth + num_levels)
        update_defaults(max_depth=max_depth)

        for _ in range(num_levels):
            [l.partition() for l in leaves if l.depth == old_depth and l.partitionable()]
            leaves: List[Cluster] = list(self.root.leaves())
            old_depth = max(l.depth for l in leaves)

        return

    def search(self, query, radius):
        """ Searches the clusters for all points within radius of query.
        """
        return search(self.root, Query(point=query, radius=radius))

    def knn_search(self, query, k):
        """ Searches the clusters for the k-nearest points to the query.
        """
        return knn_search(self.root, Query(point=query, k=k))

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

    def write(self, filename):
        """ Writes the CHESS object to the given filename.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return

    @staticmethod
    def load(filename):
        """ Loads the CHESS object from the given file.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
