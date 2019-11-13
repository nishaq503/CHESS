import unittest

from chess.cluster import *


class TestCluster(unittest.TestCase):
    def test_sample_random_data(self):
        c = Cluster(np.random.randn(100, 100), 'euclidean')
        self.assertEqual(len(c._sample()), len(c._sample_unique()))

    def test_sample_duplicate_data(self):
        data = np.random.randn(50, 100)
        data = np.concatenate([data, data])
        c = Cluster(data, 'euclidean')
        self.assertEqual(len(c._sample()) / 2, len(c._sample_unique()))
