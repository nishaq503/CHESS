import unittest

from chess.cluster import *


class TestCluster(unittest.TestCase):

    def setUp(self) -> None:
        self.data = np.concatenate([np.random.randn(50, 100) - 10, np.random.randn(50, 100) + 10])
        self.c = Cluster(self.data, 'euclidean')
        return

    def test_init(self):
        with self.assertRaises(ValueError):
            Cluster(self.data, 'dodo bird')
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
        defaults.BATCH_SIZE = 10
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

    def test_all_same(self):
        data = np.ones((100, 100))
        c = Cluster(data, 'euclidean')
        with self.assertRaises(RuntimeError):
            c.partition()
        self.assertIsNone(c.left)
        self.assertIsNone(c.right)
        return

    # noinspection PyTypeChecker
    def test_contains(self):
        data = np.random.randn(100, 100) + 100
        self.assertFalse(Query(data[0], radius=0.0) in self.c)
        self.assertTrue(Query(self.data[0], radius=0.0) in self.c)
        return

    def test_str(self):
        s = str(self.c)
        self.assertIn(self.c.name, s)
        return

    def test_repr(self):
        s = repr(self.c)
        self.assertIn(str(self.c.name), s)
        return

    def test_eq(self):
        self.assertEqual(self.c, self.c)
        self.c.make_tree(
            max_depth=defaults.MAX_DEPTH,
            min_points=defaults.MIN_POINTS,
            min_radius=defaults.MIN_RADIUS,
            stopping_criteria=None
        )
        self.assertNotEqual(self.c.left, self.c.right)
        return

    def test_dict(self):
        self.c.make_tree(
            max_depth=defaults.MAX_DEPTH,
            min_points=defaults.MIN_POINTS,
            min_radius=defaults.MIN_RADIUS,
            stopping_criteria=None
        )
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
        self.c.make_tree(
            max_depth=defaults.MAX_DEPTH,
            min_points=defaults.MIN_POINTS,
            min_radius=defaults.MIN_RADIUS,
            stopping_criteria=None
        )
        [self.assertFalse(
            c.partitionable(
                defaults.MAX_DEPTH,
                defaults.MIN_POINTS,
                defaults.MIN_RADIUS,
                stopping_criteria=None
            )
        ) for c in self.c.leaves()]
        return

    def test_partition(self):
        self.c.partition()
        self.assertTrue(self.c.left and self.c.right)
        self.assertEqual(len(self.c.left), len(self.c.right))
        return

    def test_make_tree(self):
        self.assertFalse(self.c.left or self.c.right)
        self.c.make_tree(
            max_depth=defaults.MAX_DEPTH,
            min_points=defaults.MIN_POINTS,
            min_radius=defaults.MIN_RADIUS,
            stopping_criteria=None
        )
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

        c.left.partition(), c.right.partition()
        clusters = list(c.leaves())
        self.assertEqual(len(clusters), 4)
        self.assertListEqual(['00', '01', '10', '11'], [c.name for c in clusters])

        clusters = list(c.leaves(0))
        self.assertEqual(len(clusters), 1)
        self.assertListEqual([''], [c.name for c in clusters])

        clusters = list(c.leaves(1))
        self.assertEqual(len(clusters), 2)
        self.assertListEqual(['0', '1'], [c.name for c in clusters])

        clusters = list(c.leaves(2))
        self.assertEqual(len(clusters), 4)
        self.assertListEqual(['00', '01', '10', '11'], [c.name for c in clusters])
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

    # noinspection DuplicatedCode
    def _create_ring_data(self):
        np.random.seed(42)
        scale, num_points = 12, 10_000
        samples: np.ndarray = scale * (np.random.rand(2, num_points) - 0.5)
        distances = np.linalg.norm(samples, axis=0)
        x = [samples[0, i] for i in range(num_points)
             if distances[i] < 6 and (distances[i] > 4 or distances[i] < 2)]
        y = [samples[1, i] for i in range(num_points)
             if distances[i] < 6 and (distances[i] > 4 or distances[i] < 2)]
        self.data: np.ndarray = np.asarray((x, y)).T

        self.labels = [0 if d < 2 else 1 for d in distances if d < 6 and (d > 4 or d < 2)]
        self.weights = {k: v / len(self.labels) for k, v in dict(Counter(self.labels)).items()}

        self.c = Cluster(data=self.data, metric='euclidean')
        return

    def test_classify_cluster(self):
        self._create_ring_data()
        self.c.class_distribution(data_labels=self.labels, data_weights=self.weights)
        self.assertSetEqual(set(self.weights.keys()), set(self.c.classification.keys()))
        self.assertTrue(all([(1. / len(self.weights.keys()) == v) for v in self.c.classification.values()]))
        return

    def test_json(self):
        data = np.random.randn(100, 100)
        # Testing 1 level deep.
        original = Cluster(data, 'euclidean')
        d = json.loads(original.json())
        self.assertIn('points', d)
        self.assertFalse(d['left'])
        self.assertFalse(d['right'])
        # Reloading 1 level
        loaded = Cluster.from_json(original.json(), data)
        self.assertEqual(original, loaded)

        # Full Tree.
        original.make_tree(max_depth=np.inf, min_points=1, min_radius=0.0, stopping_criteria=None)
        loaded = Cluster.from_json(original.json(), data)
        self.assertEqual(original, loaded)
        self.assertEqual(original.left, loaded.left)
        return
