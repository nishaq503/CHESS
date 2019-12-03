import unittest

import numpy as np

from chess.search import *


# noinspection PyTypeChecker
class TestSearch(unittest.TestCase):
    def test_cluster_search(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean', points=list(range(data.shape[0])), name='')
        clusters = cluster_search(c, Query(point=data[0], radius=0.1))
        self.assertEqual(len(clusters), 1)

        c.partition()
        clusters = cluster_search(c, Query(point=data[0], radius=0.1))
        self.assertEqual(len(clusters), 2)
        return

    def test_linear_search(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean', points=list(range(data.shape[0])), name='')

        points = linear_search(c, Query(data[0], radius=0.0))
        self.assertEqual(len(points), 1)

        points = linear_search(c, Query(data[0], radius=100_000))
        self.assertEqual(len(points), 100)

        points = linear_search(c, Query(data[0] + 100_000, radius=0.0))
        self.assertEqual(len(points), 0)
        return

    def test_search(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean', points=list(range(data.shape[0])), name='')
        c.partition()
        cluster_result = search(c, Query(point=data[0], radius=0.1))
        linear_result = linear_search(c, Query(point=data[0], radius=0.1))
        self.assertEqual(len(linear_result), len(cluster_result))
