import unittest
from tempfile import TemporaryFile

import chess.criterion as criterion
import chess.datasets as datasets
from chess.manifold import *

np.random.seed(42)


class TestManifold(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, cls.labels = datasets.random()
        cls.manifold = Manifold(cls.data, 'euclidean').build()
        return

    def test_init(self):
        m = Manifold(self.data, 'euclidean')
        self.assertEqual(1, len(m.graphs))

        m = Manifold(self.data, 'euclidean', [1, 2, 3])
        self.assertListEqual([1, 2, 3], m.argpoints)

        m = Manifold(self.data, 'euclidean', 0.2)
        self.assertEqual(20, len(m.argpoints))

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            Manifold(self.data, 'euclidean', ['a', 'b', 'c'])

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            Manifold(self.data, 'euclidean', 'apples')
        return

    def test_eq(self):
        self.assertEqual(self.manifold, self.manifold)
        other = Manifold(self.data, 'euclidean', argpoints=0.2)
        self.assertNotEqual(self.manifold, other)
        other.build()
        self.assertNotEqual(self.manifold, other)
        self.assertEqual(other, other)
        return

    def test_getitem(self):
        self.assertEqual(self.manifold.graphs[0], self.manifold[0])
        with self.assertRaises(IndexError):
            _ = self.manifold[100]
        return

    def test_subgraph(self):
        g = self.manifold.graphs[-1]
        c = next(iter(g))
        self.assertIn(self.manifold.subgraph(c), g.subgraphs)
        self.assertIn(self.manifold.subgraph(c.name), g.subgraphs)
        return

    def test_graph(self):
        g = self.manifold.graphs[-1]
        c = next(iter(g))
        self.assertEqual(g, self.manifold.graph(c))
        self.assertEqual(g, self.manifold.graph(c.name))
        return

    def test_iter(self):
        self.assertListEqual(self.manifold.graphs, list(iter(self.manifold)))
        return

    def test_str(self):
        self.assertIsInstance(str(self.manifold), str)
        return

    def test_repr(self):
        self.assertIsInstance(repr(self.manifold), str)
        return

    def test_find_points(self):
        self.assertEqual(1, len(self.manifold.find_points(self.data[0], radius=0.0)))
        self.assertGreaterEqual(1, len(self.manifold.find_points(self.data[0], radius=1.0)))
        return

    def test_find_clusters(self):
        self.assertEqual(1, len(self.manifold.find_clusters(self.data[0], radius=0.0, depth=-1)))
        return

    def test_build(self):
        m = Manifold(self.data, 'euclidean').build(criterion.MaxDepth(1))
        self.assertEqual(2, len(m.graphs))
        m.build(criterion.MaxDepth(2))
        self.assertEqual(3, len(m.graphs))
        m.build()
        self.assertEqual(len(self.data), len(m.graphs[-1]))
        return

    def test_build_tree(self):
        m = Manifold(self.data, 'euclidean')
        self.assertEqual(1, len(m.graphs))

        m.build_tree(criterion.AddLevels(2))
        self.assertEqual(3, len(m.graphs))

        # MaxDepth shouldn't do anything in build_tree if we're beyond that depth already.
        m.build_tree(criterion.MaxDepth(1))
        self.assertEqual(3, len(m.graphs))

        m.build_tree()
        self.assertEqual(len(self.data), len(m.graphs[-1]))
        return

    def test_select(self):
        cluster = None
        for cluster in self.manifold.graphs[-1]:
            self.assertIsInstance(self.manifold.select(cluster.name), Cluster)
        else:
            with self.assertRaises(AssertionError):
                self.manifold.select(cluster.name + '1')
        return

    def test_dump(self):
        with TemporaryFile() as fp:
            self.manifold.dump(fp)
        return

    def test_load(self):
        original = Manifold(self.data, 'euclidean').build(criterion.MinPoints(len(self.data) / 10))
        with TemporaryFile() as fp:
            original.dump(fp)
            fp.seek(0)
            loaded = Manifold.load(fp, self.data)
        self.assertEqual(original, loaded)
        self.assertEqual(original[0], loaded[0])
        return

    def test_partition_backends(self):
        data = datasets.random(n=100, dimensions=5)[0]
        m_single = Manifold(data, 'euclidean')._partition_single([criterion.MaxDepth(5)])
        m_thread = Manifold(data, 'euclidean')._partition_threaded([criterion.MaxDepth(5)])
        self.assertEqual(m_single, m_thread)
