import unittest

from chess.cluster import *


class TestCluster(unittest.TestCase):

    def setUp(self) -> None:
        self.data = np.concatenate([np.random.randn(50, 100) - 10, np.random.randn(50, 100) + 10])
        self.c = Cluster(self.data, 'euclidean')
        return

    def test_init(self):
        with self.assertRaises(ValueError):
            Cluster(self.data, 'boopscooparoop')
        c = Cluster(self.data, 'euclidean')
        self.assertEqual(c.data.shape, self.data.shape)
        self.assertEqual(c.metric, 'euclidean')
        self.assertEqual(c.name, '')
        self.assertEqual(len(c.points), self.data.shape[0])
        self.assertEqual(len(c.points), len(set(c.points)))
        self.assertFalse(c.subsample)
        self.assertEqual(len(c.samples), 100)
        self.assertEqual(c.distances.shape, (100, 100))
        self.assertTrue(c.center is not None)
        self.assertEqual(c.depth, 0)
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

    def test_getitem(self):
        data = np.random.randn(100, 100)
        c = Cluster(data, 'euclidean')
        self.assertTrue(np.all(np.equal(c[0], data[0])))
        return

    def test_contains(self):
        data = np.random.randn(100, 100) + 100
        self.assertFalse(Query(data[0], radius=0.0) in self.c)
        self.assertTrue(Query(self.data[0], radius=0.0) in self.c)
        return

    def test_str(self):
        s = str(self.c)
        for p in self.c.points:
            self.assertIn(str(p), s)
        return

    def test_repr(self):
        s = repr(self.c)
        self.assertIn(str(self.c.partitionable()), s)
        self.assertIn(str(self.c.name), s)
        return

    def test_eq(self):
        self.assertEqual(self.c, self.c)
        self.c.make_tree()
        self.assertNotEqual(self.c.left, self.c.right)
        return

    def test_dict(self):
        self.c.make_tree()
        d = self.c.dict()
        self.assertEqual(len(d.keys()), len([c for c in self.c.inorder()]))
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

    def test_partitionable(self):
        self.assertTrue(self.c.partitionable())
        self.c.make_tree()
        [self.assertFalse(c.partitionable()) for c in self.c.leaves()]
        return

    def test_partition(self):
        self.c.partition()
        self.assertTrue(self.c.left and self.c.right)
        self.assertEqual(len(self.c.left), len(self.c.right))
        return

    def test_make_tree(self):
        self.assertFalse(self.c.left or self.c.right)
        self.c.make_tree()
        self.assertGreater(len([c for c in self.c.leaves()]), 2)
        return

    def test_compress(self):
        points = self.c.compress()
        self.assertEqual(len(points), self.data.shape[0])
        self.assertEqual(len(points[0]), self.data.shape[1])
        return

    def test_inorder(self):
        data = np.random.randn(50, 100)
        c = Cluster(np.concatenate([data - 10, data + 10]), 'euclidean')
        c.partition()
        clusters = list(c.inorder())
        self.assertEqual(len(clusters), 3)
        self.assertListEqual(['0', '', '1'], [c.name for c in clusters])
        return

    def test_postorder(self):
        self.c.partition()
        clusters = list(self.c.postorder())
        self.assertEqual(len(clusters), 3)
        self.assertListEqual(['0', '1', ''], [c.name for c in clusters])
        return

    def test_preorder(self):
        self.c.partition()
        clusters = list(self.c.preorder())
        self.assertEqual(len(clusters), 3)
        self.assertListEqual(['', '0', '1'], [c.name for c in clusters])
        return

    def test_leaves(self):
        data = np.random.randn(50, 100)
        c = Cluster(np.concatenate([data - 10, data + 10]), 'euclidean')
        c.partition()
        clusters = list(c.leaves())
        self.assertEqual(len(clusters), 2)
        self.assertListEqual(['0', '1'], [c.name for c in clusters])
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

    def test_sample_random_data(self):
        c = Cluster(np.random.randn(100, 100), 'euclidean')
        self.assertEqual(len(c.samples), 100)
        return
