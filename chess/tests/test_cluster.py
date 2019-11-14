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

    def test_pairwise_distances(self):
        data = np.random.randn(50, 100) + 1
        c1 = Cluster(data, 'euclidean')
        distances = c1._pairwise_distances()
        self.assertEqual(distances.shape, (50, 50))
        self.assertGreater(distances.sum(), 0)
        self.assertEqual(np.diagonal(distances).sum().round(), 0)

        c2 = Cluster(np.concatenate((data, data)), 'euclidean')
        distances = c1._pairwise_distances()
        self.assertEqual(distances.shape, (50, 50))
        self.assertGreater(distances.sum(), 0)
        self.assertEqual(np.diagonal(distances).sum().round(), 0)

    def test_iter_batch(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean')
        iter = c._iter_batch(30)
        self.assertEqual(len(next(iter)), 30)
        self.assertEqual(len(next(iter)), 30)
        self.assertEqual(len(next(iter)), 30)
        self.assertEqual(len(next(iter)), 10)
        return

    def test_radius(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean')
        with self.assertRaises(NotImplementedError):
            c.radius()
        return
