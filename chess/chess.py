""" CHESS API

This class wraps the underlying Cluster structure with a convenient API.
"""
import pickle

import numpy as np

from .cluster import Cluster
from .knn_search import knn_search
from .query import Query
from .search import search


class CHESS:
    """ Clustered Hierarchical Entropy-Scaling Search Object.
    """

    def __init__(self, data: np.memmap, metric: str):
        self.cluster = Cluster(data, metric)
        self.metric = metric

    def __str__(self):
        """
        :return: CSV-style string with two columns, name and an array of points, one row per leaf cluster.
        """
        return '\n'.join([
            'name, points',
            *[str(c) for c in self.cluster.leaves()]
        ])

    def __repr__(self):
        """
        :return: CSV-style string with more attributes, one row per cluster, generated inorder.
        """
        return '\n'.join([
            'name, number_of_points, center, radius, lfd, is_leaf',
            *[repr(c) for c in self.cluster.inorder()]
        ])

    def __eq__(self, other):
        return self.metric == other.metric and self.cluster == other.cluster

    def build(self, stopping_criteria=None):
        """ Clusters points recursively until stopping_criteria returns True.

        :param stopping_criteria: optional override to cluster.partitionable
        """
        self.cluster.make_tree(stopping_criteria)
        return

    def search(self, query, radius):
        """ Searches the clusters for all points within radius of query.
        """
        return search(self.cluster, Query(point=query, radius=radius))

    def knn_search(self, query, k):
        """ Searches the clusters for the k-nearest points to the query.
        """
        return knn_search(self.cluster, Query(point=query, k=k))

    def compress(self, filename: str):
        """ Compresses the clusters.
        """
        mm = np.memmap(
            filename,
            dtype=self.cluster.data.dtype,
            mode='w+',
            shape=self.cluster.data.shape,
        )
        i = 0
        for cluster in self.cluster.leaves():
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
