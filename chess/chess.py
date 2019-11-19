""" CHESS API

This class wraps the underlying Cluster structure with a convenient API.
"""
import pickle
from typing import Callable

import numpy as np

from chess import defaults
from .cluster import Cluster
from .knn_search import knn_search
from .query import Query
from .search import search


class CHESS:
    """ Clustered Hierarchical Entropy-Scaling Search Object.
    """

    def __init__(
            self,
            data: np.memmap,
            metric: str,
            max_depth: int = defaults.MAX_DEPTH,
            min_points: int = defaults.MIN_POINTS,
            min_radius: defaults.RADII_DTYPE = defaults.MIN_RADIUS,
            stopping_criteria: Callable[[any], bool] = None,
    ):
        self.data = data
        self.metric = metric
        self.max_depth = max_depth
        self.min_points = min_points
        self.min_radius = min_radius
        self.stopping_criteria = stopping_criteria

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
            stopping_criteria: Callable[[any], bool] = None,
    ):
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
