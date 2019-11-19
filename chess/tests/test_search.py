import unittest

import numpy as np

from chess import defaults
from chess.search import *


class TestSearch(unittest.TestCase):
    def test_cluster_search(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean')
        clusters = cluster_search(c, Query(point=data[0], radius=defaults.RADII_DTYPE(0.0)))
        self.assertEqual(len(clusters), 1)
