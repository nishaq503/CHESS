import unittest

from chess import criterion
from chess.manifold import *

np.random.seed(42)


def linear_search(point: Data, radius: Radius, data: Data, metric: str):
    point = np.expand_dims(point, 0)
    results = []
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i: i + BATCH_SIZE]
        distances = cdist(point, batch, metric)[0]
        results.extend([p for p, d in zip(batch, distances) if d <= radius])
    return results


class TestManifoldFunctional(unittest.TestCase):
    def test_random(self):
        # We begin by getting some data and building with no constraints.
        data = np.random.randn(1000, 3)
        m = Manifold(data, 'euclidean')
        m.build()
        # With no constraints, clusters should be singletons.
        self.assertEqual(1, len(m.find_clusters(data[0], 0.0, -1)))
        self.assertEqual(1, len(m.find_points(data[0], radius=0.0)))

        m.build(criterion.MinPoints(10))
        c = next(iter(m.find_clusters(data[0], 0.0, -1)))
        self.assertEqual(len(linear_search(c.center, c.radius, data, m.metric)), len(m.find_points(c.center, c.radius)))
        return

    def test_all_same(self):
        # A bit simpler, every point is the same.
        data = np.ones((1000, 3))
        m = Manifold(data, 'euclidean')
        m.build()
        # There should only ever be one cluster here.
        self.assertEqual(1, len(m.graphs))
        m.deepen()
        # Even after explicit deepen calls.
        self.assertEqual(1, len(m.graphs))
        self.assertEqual(1, len(m.find_clusters(data[0], 0.0, -1)))
        # And, we should get all 1000 points back for any of the data.
        self.assertEqual(1000, len(m.find_points(data[0], 0.0)))
        return

    def test_two_clumps(self):
        # Here we have two distinct clusters.
        data = np.concatenate([np.ones((500, 2)) * -2, np.ones((500, 2)) * 2])
        m = Manifold(data, 'euclidean')
        # We expect building to stop with two clusters.
        m.build()
        self.assertEqual(2, len(m.graphs[-1]))
        return


class TestManifold(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.random.randn(100, 100)
        return

    def test_init(self):
        Manifold(self.data, 'euclidean')

        m = Manifold(self.data, 'euclidean', [1, 2, 3])
        self.assertListEqual([1, 2, 3], m.argpoints)

        m = Manifold(self.data, 'euclidean', 0.2)
        self.assertEqual(20, len(m.argpoints))

        with self.assertRaises(ValueError):
            Manifold(self.data, 'euclidean', ['a', 'b', 'c'])
        return

    def test_eq(self):
        m1 = Manifold(self.data, 'euclidean')
        m2 = Manifold(self.data, 'euclidean')
        self.assertEqual(m1, m2)
        m2.metric = 'boop'
        self.assertNotEqual(m1, m2)
        return

    def test_getitem(self):
        m = Manifold(self.data, 'euclidean')
        self.assertEqual(m[0], m.graphs[0])
        return

    def test_iter(self):
        m = Manifold(self.data, 'euclidean')
        self.assertListEqual(m.graphs, list(iter(m)))
        return

    def test_str(self):
        m = Manifold(self.data, 'euclidean')
        self.assertIsInstance(str(m), str)
        return

    def test_repr(self):
        m = Manifold(self.data, 'euclidean')
        # TODO
        self.assertIsInstance(repr(m), str)
        return
