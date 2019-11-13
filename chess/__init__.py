import numpy as np

from chess.cluster import Cluster


class CHESS:

    def __init__(self, data: np.memmap, metric: str):
        self.data = Cluster(data, metric)
        self.metric = metric
        self.cluster = None

    def cluster(self, stopping_criteria):
        pass

    def search(self, query, radius):
        pass

    def knn_search(self, query, k):
        pass

    def compress(self):
        pass

    def write(self, filename):
        pass

    @staticmethod
    def load(filename):
        pass
