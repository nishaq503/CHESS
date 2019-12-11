import unittest

from chess.criterion import *
from chess.datasets import *
from chess.manifold import *

MIN_RADIUS = 0.5


class TestCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.random.randn(100, 100)
        cls.manifold = Manifold(cls.data, 'euclidean')
        cls.all_zeros = np.zeros((100, 100))
        cls.cluster = Cluster(cls.manifold, cls.manifold.argpoints, '')
        return

    def test_init(self):
        Cluster(self.manifold, self.manifold.argpoints, '')
        with self.assertRaises(ValueError):
            Cluster(self.manifold, [], '')
        return

    def test_points(self):
        self.assertTrue(np.array_equal(
            self.manifold.data,
            self.cluster.points
        ))
        return

    def test_argpoints(self):
        self.assertSetEqual(
            set(self.manifold.argpoints),
            set(self.cluster.argpoints)
        )
        return

    def test_samples(self):
        pass

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
            np.sqrt(len(self.data)),
            self.cluster.nsamples
        )

    def test_overlaps(self):
        point = np.ones((100, ))
        self.assertTrue(self.cluster.overlaps(point, 1.))

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
        return

    @staticmethod
    def trace_lineage(cluster: Cluster, other: Cluster):  # TODO: Cover
        assert cluster.depth == other.depth
        assert cluster.overlaps(other.center, other.radius)
        lineage = [other.name[:i] for i in range(other.depth) if cluster.name[:i] != other.name[:i]]
        ancestors = [other.manifold.select(n) for n in reversed(lineage)]
        for ancestor in ancestors:
            print(f'checking {ancestor.name}...')
            if not cluster.overlaps(ancestor.center, 2 * ancestor.radius):
                print(f'{cluster.name} did not overlap with {ancestor.name}')
                distance = cluster.distance(np.asarray([ancestor.center], dtype=np.float64))[0]
                print(f'cluster.radius: {cluster.radius} vs ancestor.radius: {ancestor.radius}')
                print(f'distance: {distance} vs cluster.radius + 2 * ancestor.radius: {cluster.radius + 2 * ancestor.radius}')
                print(f'off by {(distance - (cluster.radius + 2 * ancestor.radius)) / distance} percent')
                print(f'cluster.depth: {cluster.depth} vs ancestor.depth: {ancestor.depth}')
                print(f'cluster_population: {len(cluster.argpoints)} vs ancestor_population: {len(ancestor.argpoints)}')
                print('\n\n\n')
                return
        else:
            raise ValueError(f'all divergent ancestors had overlap')

    def test_neighbors_more(self):
        data, labels = bullseye()
        np.random.seed(42)
        manifold = Manifold(data, 'euclidean')
        manifold.build(MinRadius(MIN_RADIUS), MaxDepth(12))
        for depth, graph in enumerate(manifold.graphs):
            for cluster in graph:
                neighbors = manifold.find_clusters(cluster.center, cluster.radius, depth) - {cluster}
                if (neighbors - set(cluster.neighbors.keys())) or (set(cluster.neighbors.keys()) - neighbors):
                    print(depth, cluster.name, ':', [n.name for n in neighbors])
                    print(depth, cluster.name, ':', [n.name for n in (neighbors - set(cluster.neighbors.keys()))])
                    print(depth, cluster.name, ':', [n.name for n in (set(cluster.neighbors.keys()) - neighbors)])
                # self.assertTrue(len(neighbors) >= len(set(cluster.neighbors.keys())))
                self.assertSetEqual(neighbors, set(cluster.neighbors.keys()))

    def test_tree_search(self):
        data, labels = bullseye()
        m = Manifold(data, 'euclidean')
        m.build(MinRadius(MIN_RADIUS), MaxDepth(12))
        for depth, graph in enumerate(m.graphs):
            for cluster in graph:
                linear = set([c for c in graph if c.overlaps(cluster.center, cluster.radius)])
                tree = set(next(iter(m.graphs[0])).tree_search(cluster.center, cluster.radius, cluster.depth))
                print(depth, [c.name for c in (linear ^ tree)])
                for d in range(depth, 0, -1):
                    parents = set([m.select(cluster.name[:-1]) for cluster in linear])
                    for parent in parents:
                        self.assertIn(parent, parent.tree_search(cluster.center, cluster.radius, parent.depth))
                # self.assertSetEqual(linear, tree, f"Sets unequal for cluster: {cluster.name}")
