import numpy as np

from chess.cluster import Cluster
from chess.knn_search import knn_search
from chess.query import Query
from chess.search import search


class CHESS:

    def __init__(self, data: np.memmap, metric: str):
        self.cluster = Cluster(data, metric)
        self.metric = metric

    def __str__(self):
        return '\n'.join([str(c) for c in self.cluster.inorder()])

    def __repr__(self):
        return '\n'.join([repr(c) for c in self.cluster.inorder()])

    def cluster(self, stopping_criteria):
        """ Clusters points recursively until stopping_criteria returns True. """
        raise NotImplementedError

    def search(self, query, radius):
        """ Searches the clusters for all points within query.radius of query.point.
        """
        return search(self.cluster, Query(point=query, radius=radius))

    def knn_search(self, query, k):
        return knn_search(self.cluster, Query(point=query, k=k))

    def compress(self):
        """ Compresses the clusters. """
        raise NotImplementedError

    def write(self, filename):
        """ Writes the CHESS object to the given filename. """
        raise NotImplementedError

    @staticmethod
    def load(filename):
        """ Loads the CHESS object from the given file. """
        raise NotImplementedError
