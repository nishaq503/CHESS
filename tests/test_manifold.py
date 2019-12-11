import os
import tempfile
import unittest

from chess.criterion import *
from chess.manifold import *

np.random.seed(42)


def linear_search(point: Data, radius: Radius, data: Data, metric: str):
    point = np.expand_dims(point, 0)
    results, argresults = list(), list()
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i: i + BATCH_SIZE]
        distances = cdist(point, batch, metric)[0]
        results.extend([p for p, d in zip(batch, distances) if d <= radius])
        argresults.extend([j for j, d in zip(range(i, min(i + BATCH_SIZE, data.shape[0])), distances) if d <= radius])
    return results, argresults


class TestManifoldFunctional(unittest.TestCase):
    def test_random(self):
        # We begin by getting some data and building with no constraints.
        data = np.random.randn(100, 100)
        m = Manifold(data, 'euclidean')
        m.build()
        # With no constraints, clusters should be singletons.
        self.assertEqual(1, len(m.find_clusters(data[0], 0., -1)))
        self.assertEqual(1, len(m.find_points(data[0], 0.)))

        data = np.random.randn(1000, 100)
        m = Manifold(data, 'euclidean')
        m.build(MinPoints(10))
        for _ in range(10):
            point = int(np.random.choice(1000))
            linear_results, linear_argresults = linear_search(data[point], 0.5, data, m.metric)
            self.assertTrue(1 <= len(linear_argresults))
            self.assertIn(point, linear_argresults)
            self.assertEqual(len(linear_results), len(m.find_points(data[point], 0.5, mode='iterative')))
            self.assertEqual(len(linear_results), len(m.find_points(data[point], 0.5, mode='recursive')))
            self.assertEqual(len(linear_results), len(m.find_points(data[point], 0.5, mode='dfs')))
            self.assertEqual(len(linear_results), len(m.find_points(data[point], 0.5, mode='bfs')))
            # self.assertEqual(len(linear_search(data[point], 0.5, data, m.metric)), len(m.find_points(data[point], 0.5, mode='recursive')))
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

    def test_radius_decreasing(self):
        data = np.random.random(size=(100, 20))
        manifold = Manifold(data, 'euclidean', new_calculate_neighbors=True)
        manifold.build(MaxDepth(6))

        for graph in manifold.graphs:
            for cluster in graph.clusters:
                # print(f'checking {cluster.name}')
                ancestors = [manifold.select(cluster.name[:i]) for i in range(cluster.depth)]
                radii_differences = [ancestors[i-1].radius - ancestors[i].radius for i in range(1, len(ancestors))]
                # if len(radii_differences) > 0:
                #     print(f'\n{[a.name for a in ancestors]}\n{[a.radius for a in ancestors]}\n{radii_differences}')
                #     self.assertTrue(all((r >= 0. for r in radii_differences)),
                #                     msg=f'\n{[a.name for a in ancestors]}\n{radii_differences}')
        return

    def test_two_points_with_dups(self):
        # Here we have two distinct clusters.
        data = np.concatenate([np.ones((500, 2)) * -2, np.ones((500, 2)) * 2])
        m = Manifold(data, 'euclidean')
        # We expect building to stop with two clusters.
        m.build()
        self.assertEqual(2, len(m.graphs[-1]))
        return

    def test_two_clumps(self):  # TODO: Tom
        data = np.concatenate([np.random.randn(50, 2) * -5, np.random.randn(50, 2) * 5])
        m = Manifold(data, 'euclidean')
        m.build(MinNeighborhood(starting_depth=1, threshold=1))
        # self.assertEqual(2, len(m.graphs))
        # self.assertEqual(1, len(next(iter(m.graphs[-1])).neighbors))
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
            # noinspection PyTypeChecker
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

    # noinspection DuplicatedCode
    def test_json(self):
        original = Manifold(self.data, 'euclidean')
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'dump'), 'w') as outfile:
                original.json_dump(outfile)
            with open(os.path.join(d, 'dump'), 'r') as infile:
                loaded: Manifold = Manifold.json_load(infile, self.data)
        self.assertEqual(original, loaded)

        original.build()
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'dump'), 'w') as outfile:
                original.json_dump(outfile)
            with open(os.path.join(d, 'dump'), 'r') as infile:
                loaded: Manifold = Manifold.json_load(infile, self.data)
        self.assertEqual(len(original.graphs), len(loaded.graphs))
        self.assertEqual(original, loaded)

    # noinspection DuplicatedCode
    def test_pickle(self):
        original = Manifold(self.data, 'euclidean')
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'dump'), 'wb') as outfile:
                original.pickle_dump(outfile)
            with open(os.path.join(d, 'dump'), 'rb') as infile:
                loaded: Manifold = Manifold.pickle_load(infile, self.data)
        self.assertEqual(original, loaded)

        original.build()
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'dump'), 'wb') as outfile:
                original.pickle_dump(outfile)
            with open(os.path.join(d, 'dump'), 'rb') as infile:
                loaded: Manifold = Manifold.pickle_load(infile, self.data)
        self.assertEqual(len(original.graphs), len(loaded.graphs))
        self.assertEqual(original, loaded)
