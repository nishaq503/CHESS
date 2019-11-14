import unittest

from chess.cluster import *


class TestCluster(unittest.TestCase):

    def setUp(self) -> None:
        self.data = np.concatenate([np.random.randn(50, 100) - 10, np.random.randn(50, 100) + 10])
        self.c = Cluster(self.data, 'euclidean')
        return

    def test_sample_random_data(self):
        c = Cluster(np.random.randn(100, 100), 'euclidean')
        self.assertEqual(len(c.samples), 100)
        return

    def test_pairwise_distances(self):
        data = np.random.randn(50, 100) + 1
        c1 = Cluster(data, 'euclidean')
        distances = c1.distances
        self.assertEqual(distances.shape, (50, 50))
        self.assertGreater(distances.sum(), 0)
        self.assertEqual(np.diagonal(distances).sum().round(), 0)

        c2 = Cluster(np.concatenate((data, data)), 'euclidean')
        distances = c2.distances
        self.assertEqual(distances.shape, (100, 100))
        self.assertGreater(distances.sum(), 0)
        self.assertEqual(np.diagonal(distances).sum().round(), 0)
        return

    def test_iter_batch(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean')
        iter = c._iter_batch(batch_size=30)
        self.assertEqual(len(next(iter)), 30)
        self.assertEqual(len(next(iter)), 30)
        self.assertEqual(len(next(iter)), 30)
        self.assertEqual(len(next(iter)), 10)
        return

    def test_radius(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean')
        self.assertGreater(c.radius(), 0)
        return

    def test_local_fractal_dimension(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean')
        self.assertGreater(c.local_fractal_dimension(), 0)
        return

    def test_iter(self):
        data = np.random.randn(100, 100)
        globals.BATCH_SIZE = 10
        c = Cluster(data, 'euclidean')
        d = [b for b in c]
        self.assertEqual(len(d), 10)
        self.assertTrue(np.all(np.equal(data, np.concatenate(d))))
        return

    def test_len(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean')
        self.assertEqual(len(c), 100)
        return

    def test_indexing(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean')
        self.assertTrue(np.all(np.equal(c[0], data[0])))
        return

    def test_partition(self):
        self.c.partition()
        self.assertTrue(self.c.left and self.c.right)
        self.assertEqual(len(self.c.left), len(self.c.right))
        return

    def test_inorder(self):
        data = np.random.randn(50, 100)
        c = Cluster(np.concatenate([data - 10, data + 10]), 'euclidean')
        c.partition()
        clusters = list(c.inorder())
        self.assertEqual(len(clusters), 3)
        return

    def test_leaves(self):
        data = np.random.randn(50, 100)
        c = Cluster(np.concatenate([data - 10, data + 10]), 'euclidean')
        c.partition()
        clusters = list(c.leaves())
        self.assertEqual(len(clusters), 2)
