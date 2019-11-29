import os
import tempfile
import unittest

from chess.chess import *


# noinspection PyTypeChecker
class TestCHESS(unittest.TestCase):
    tempfile: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempfile = tempfile.NamedTemporaryFile()
        data = np.random.randn(1000, 100)
        cls.data = np.memmap(cls.tempfile, dtype='float32', mode='w+', shape=data.shape)
        cls.data[:] = data[:]
        return

    @classmethod
    def tearDownClass(cls) -> None:
        return

    def test_functional(self):
        chess = CHESS(self.data, 'euclidean')
        chess.build()
        result = chess.search(self.data[0], 0.0)
        self.assertEqual(len(result), 1)

        result = chess.search(self.data[0] + 50, 0.0)
        self.assertEqual(len(result), 0)

        result = chess.search(self.data[0], 20.0)
        self.assertEqual(len(result), 1000)
        return

    def test_init(self):
        CHESS(self.data, 'euclidean')

        with self.assertRaises(ValueError):
            CHESS(self.data, 'dodo bird')

        return

    def test_select(self):
        data = np.random.randn(100, 100)
        chess = CHESS(data, 'euclidean')
        chess.build()
        clusters = list(chess.root.inorder())

        for cluster in clusters:
            c = chess.select(cluster.name)
            self.assertEqual(c.name, cluster.name)
            self.assertEqual(c, cluster)

        with self.assertRaises(ValueError):
            chess.select('elmo')

        c = chess.select(sorted(clusters, key=lambda c_: len(c_.name), reverse=True)[0].name + '0')
        self.assertIsNone(c)
        return

    def test_graph(self):
        data = np.random.randn(100, 100)
        chess = CHESS(data, 'euclidean')
        chess.build()
        chess.graph.cache_clear()
        g1 = chess.graph(1)
        g2 = chess.graph(2)
        self.assertNotEqual(g1, g2)
        self.assertDictEqual(g2, graph(list(chess.root.leaves(2))))

        [chess.graph(1) for _ in range(100)]
        self.assertEqual(chess.graph.cache_info().misses, 2)
        return

    def test_connected_subgraph(self):
        data = np.random.randn(100, 100)
        chess = CHESS(np.concatenate([data - 1_000, data + 1_000]), 'euclidean')
        chess.build()

        leaves = list(chess.root.leaves())
        [self.assertLessEqual(len(set(chess.subgraph(c).keys())), len(leaves)) for c in leaves]
        return

    def test_all_same(self):
        data = np.ones((100, 100))
        c = CHESS(data, 'euclidean')
        c.build()
        self.assertIsNone(c.root.left)
        self.assertIsNone(c.root.right)

        c = CHESS(np.concatenate([data - 100, data + 100]), 'euclidean')
        c.build()
        self.assertIsNotNone(c.root.left)
        self.assertIsNotNone(c.root.right)
        return

    def test_str(self):
        chess = CHESS(self.data, 'euclidean')
        chess.build()
        s = str(chess)
        self.assertGreater(len(s), 0)
        # Do we have the right number of clusters? (Exclude title row)
        self.assertEqual(len(s.split('\n')[1:]), len([c for c in chess.root.leaves()]))
        return

    def test_repr(self):
        chess = CHESS(self.data, 'euclidean')
        chess.build()
        s = repr(chess)
        self.assertGreater(len(s), 0)
        return

    def test_build(self):
        chess = CHESS(self.data, 'euclidean', max_depth=5)
        chess.build()
        self.assertTrue(all((
            chess.root.left,
            chess.root.right,
            chess.root.left.left,
            chess.root.left.right,
            chess.root.right.left,
            chess.root.right.right,
        )))
        return

    # noinspection PyTypeChecker
    def test_deepen(self):
        data = np.random.randn(2_000, 100)
        chess = CHESS(data, 'euclidean')
        chess.deepen(levels=5)
        self.assertEqual(5, max(map(len, chess.root.dict().keys())))
        chess.deepen(levels=5)
        self.assertEqual(10, max(map(len, chess.root.dict().keys())))

    def test_search(self):
        chess = CHESS(self.data, 'euclidean')
        chess.build()
        self.assertEqual(len(chess.search(self.data[0], 0.0)), 1)
        self.assertEqual(len(chess.search(self.data[0] + 100, 0.0)), 0)
        self.assertGreaterEqual(len(chess.search(self.data[0] + 0.1, 10.0)), 1)
        self.assertEqual(len(chess.search(self.data[0], 100)), 1000)
        return

    def test_compress(self):
        chess = CHESS(self.data, 'euclidean')
        chess.build()
        filepath = tempfile.NamedTemporaryFile()
        chess.compress(filepath)
        data = np.memmap(filepath, mode='r+', dtype='float32', shape=self.data.shape)
        self.assertEqual(data.shape, (1000, 100))
        filepath.close()
        return

    def test_write(self):
        chess = CHESS(self.data, 'euclidean')
        with tempfile.TemporaryDirectory() as d:
            chess.write(os.path.join(d, 'dump'))
            self.assertTrue(os.path.exists(os.path.join(d, 'dump')))

    def test_load(self):
        chess = CHESS(self.data, 'euclidean')
        with tempfile.TemporaryDirectory() as d:
            chess.write(os.path.join(d, 'dump'))
            loaded = CHESS.load(os.path.join(d, 'dump'), self.data)
        self.assertEqual(chess, loaded)

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
        labels = [0 if d < 2 else 1 for d in distances if d < 6 and (d > 4 or d < 2)]

        self.chess = CHESS(
            data=self.data,
            metric='euclidean',
            max_depth=20,
            min_points=10,
            min_radius=defaults.RADII_DTYPE(0.05),
            stopping_criteria=None,
            labels=labels,
        )
        self.chess.build()
        return

    def test_label_cluster(self):
        self._create_ring_data()
        self.assertSetEqual(set(dict(Counter(self.chess.labels)).keys()), set(self.chess.weights.keys()))
        for cluster in self.chess.root.inorder():
            classification = self.chess.label_cluster(cluster=cluster)
            self.assertSetEqual(set(self.chess.weights.keys()), set(classification.keys()))
            self.assertAlmostEqual(sum(classification.values()), 1., places=10)

    def test_label_cluster_tree(self):
        self._create_ring_data()
        self.chess.label_cluster_tree()
        labels = set(self.chess.weights.keys())
        for c in self.chess.root.inorder():
            self.assertSetEqual(labels, set(c.classification.keys()))
            self.assertAlmostEqual(sum(c.classification.values()), 1., places=10)

    def test_connected_clusters(self):
        data = np.random.randn(100, 100)
        chess = CHESS(np.concatenate([data - 100, data + 100]), 'euclidean')
        chess.build()
        results = chess.connected_clusters(0)
        self.assertEqual(len(results), 1)
        results = chess.connected_clusters()
        self.assertGreater(len(results), 1)
        return
