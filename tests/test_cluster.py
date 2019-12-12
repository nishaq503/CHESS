import unittest

from chess.criterion import *
from chess.datasets import *
from chess.manifold import *

MIN_RADIUS = 0.5


class TestCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.random.randn(1_000, 100)
        cls.manifold = Manifold(cls.data, 'euclidean')
        return

    def setUp(self) -> None:
        self.cluster = Cluster(self.manifold, self.manifold.argpoints, '')
        self.children = list(self.cluster.partition())
        return

    def test_init(self):
        Cluster(self.manifold, self.manifold.argpoints, '')
        with self.assertRaises(ValueError):
            Cluster(self.manifold, [], '')
        return

    def test_eq(self):
        self.assertEqual(self.cluster, self.cluster)
        self.assertNotEqual(self.cluster, self.children[0])
        self.assertNotEqual(self.children[0], self.children[1])
        return

    def test_hash(self):
        self.assertIsInstance(hash(self.cluster), int)
        return

    def test_str(self):
        self.assertEqual('root', str(self.cluster))
        self.assertSetEqual({'1', '2'}, set([str(c) for c in self.children]))
        return

    def test_repr(self):
        self.assertIsInstance(repr(self.cluster), str)
        return

    def test_iter(self):
        self.assertEqual(
            (self.data.shape[0] / BATCH_SIZE, BATCH_SIZE, self.data.shape[1]),
            np.array(list(self.cluster.data)).shape
        )
        return

    def test_getitem(self):
        _ = self.cluster[0]
        with self.assertRaises(IndexError):
            _ = self.cluster[len(self.data) + 1]
        return

    def test_contains(self):
        self.assertIn(self.data[0], self.cluster)
        return

    def test_metric(self):
        self.assertEqual(self.manifold.metric, self.cluster.metric)
        return

    def test_depth(self):
        self.assertEqual(0, self.cluster.depth)
        self.assertEqual(1, self.children[0].depth)
        return

    def test_points(self):
        self.assertTrue(np.array_equal(
            self.manifold.data,
            np.array(list(self.cluster.data)).reshape(self.data.shape)
        ))
        return

    def test_argpoints(self):
        self.assertSetEqual(
            set(self.manifold.argpoints),
            set(self.cluster.argpoints)
        )
        return

    def test_samples(self):
        self.assertEqual((self.cluster.nsamples, self.data.shape[-1]), self.cluster.samples.shape)
        return

    def test_argsamples(self):
        data = np.zeros((100, 100))
        for i in range(10):
            data = np.concatenate([data, np.ones((1, 100)) * i], axis=0)
            manifold = Manifold(data, 'euclidean')
            cluster = Cluster(manifold, manifold.argpoints, '')
            self.assertLessEqual(i + 1, len(cluster.argsamples))
        return

    def test_nsamples(self):
        self.assertEqual(
            int(np.sqrt(len(self.data))),
            self.cluster.nsamples
        )
        return

    def test_centroid(self):
        self.assertEqual((self.data.shape[-1],), self.cluster.centroid.shape)
        self.assertFalse(np.any(self.cluster.centroid == self.data))
        return

    def test_medoid(self):
        self.assertEqual((self.data.shape[-1],), self.cluster.medoid.shape)
        self.assertTrue(np.any(self.cluster.medoid == self.data))
        return

    def test_argmedoid(self):
        self.assertIn(self.cluster.argmedoid, self.cluster.argpoints)
        return

    def test_radius(self):
        self.assertGreaterEqual(self.cluster.radius, 0.0)
        return

    def test_argradius(self):
        self.assertIn(self.cluster.argradius, self.cluster.argpoints)
        return

    def test_local_fractal_dimension(self):
        self.assertGreaterEqual(self.cluster.local_fractal_dimension, 0)
        return

    def test_clear_cache(self):
        self.cluster.clear_cache()
        self.assertNotIn('_argsamples', self.cluster.__dict__)
        return

    def test_tree_search(self):
        data, labels = bullseye()
        m = Manifold(data, 'euclidean')
        m.build(MinRadius(MIN_RADIUS), MaxDepth(12))
        for depth, graph in enumerate(m.graphs):
            for cluster in graph:
                linear = set([c for c in graph if c.overlaps(cluster.medoid, cluster.radius)])
                tree = set(next(iter(m.graphs[0])).tree_search(cluster.medoid, cluster.radius, cluster.depth))
                print(depth, [c.name for c in (linear ^ tree)])
                for d in range(depth, 0, -1):
                    parents = set([m.select(cluster.name[:-1]) for cluster in linear])
                    for parent in parents:
                        self.assertIn(parent, parent.tree_search(cluster.medoid, cluster.radius, parent.depth))
                # self.assertSetEqual(linear, tree, f"Sets unequal for cluster: {cluster.name}")

    def test_prune(self):
        self.cluster.prune()
        self.assertFalse(self.cluster.children)
        return

    def test_partition(self):
        children = list(self.cluster.partition())
        self.assertGreater(len(children), 1)
        return

    def test_neighbors(self):
        data = np.concatenate([np.random.randn(1000, 2) * -1, np.random.randn(1000, 2) * 1])
        m = Manifold(data, 'euclidean').build()
        parent = Cluster(m, m.argpoints, '')
        children = parent.partition()
        [self.assertNotIn(c, c.neighbors) for c in children]
        [self.assertEqual(parent.depth + 1, c.depth) for c in children]

        for i in range(2, 10):
            # noinspection PyUnresolvedReferences
            children = [c for C in children for c in C.partition()]
            [self.assertNotIn(c, c.neighbors.keys()) for c in children]
            [self.assertEqual(parent.depth + i, c.depth) for c in children]

        data, labels = bullseye()
        np.random.seed(42)
        manifold = Manifold(data, 'euclidean')
        manifold.build(MinRadius(MIN_RADIUS), MaxDepth(12))
        for depth, graph in enumerate(manifold.graphs):
            for cluster in graph:
                neighbors = manifold.find_clusters(cluster.medoid, cluster.radius, depth) - {cluster}
                if (neighbors - set(cluster.neighbors.keys())) or (set(cluster.neighbors.keys()) - neighbors):
                    print(depth, cluster.name, ':', [n.name for n in neighbors])
                    print(depth, cluster.name, ':', [n.name for n in (neighbors - set(cluster.neighbors.keys()))])
                    print(depth, cluster.name, ':', [n.name for n in (set(cluster.neighbors.keys()) - neighbors)])
                # self.assertTrue(len(neighbors) >= len(set(cluster.neighbors.keys())))
                self.assertSetEqual(neighbors, set(cluster.neighbors.keys()))
        return

    def test_distance(self):
        self.assertGreater(self.children[0].distance(np.expand_dims(self.children[1].medoid, 0)), 0)
        return

    def test_overlaps(self):
        point = np.ones((100,))
        self.assertTrue(self.cluster.overlaps(point, 1.))
        return

    def test_to_json(self):
        data = self.cluster.json()
        self.assertFalse(data['argpoints'])
        self.assertTrue(data['children'])
        data = self.children[0].json()
        self.assertTrue(data['argpoints'])
        self.assertFalse(data['children'])
        return

    def test_from_json(self):
        c = Cluster.from_json(self.manifold, self.cluster.json())
        self.assertEqual(self.cluster, c)
        return
