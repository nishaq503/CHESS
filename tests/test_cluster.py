import unittest

from chess.manifold import *
from chess.datasets import *
from chess.criterion import *


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
        self.assertListEqual(
            self.manifold.argpoints,
            self.cluster.argpoints
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
        m = Manifold(data, 'euclidean')
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
    def trace_lineage(left: Cluster, right: Cluster):
        assert left.depth == right.depth
        assert left.overlaps(right.center, right.radius)
        left_lineage = [left.name[:i] for i in range(left.depth)]
        right_lineage = [right.name[:i] for i in range(right.depth)]
        for l_, r_ in zip(left_lineage, right_lineage):
            if l_ == r_:
                continue
            else:
                left_ancestor = left.manifold.select(l_)
                right_ancestor = right.manifold.select(r_)
                if not left_ancestor.overlaps(right_ancestor.center, right_ancestor.radius):
                    print(f'{l_} and {r_} do not have overlap but their descendents {left.name} and {right.name} do.')
                    return
        return

    def test_neighbors_more(self):
        data, labels = bullseye()
        np.random.seed(42)
        manifold = Manifold(data, 'euclidean')
        manifold.build(MinRadius(MIN_RADIUS), MaxDepth(12))
        for depth, graph in enumerate(manifold.graphs):
            for cluster in graph:
                naive_neighbors = {c for c in graph if c.overlaps(cluster.center, cluster.radius)} - {cluster}
                if naive_neighbors - set(cluster.neighbors.keys()):
                    print(depth, cluster.name, ':', [n.name for n in naive_neighbors])
                    print(depth, cluster.name, 'missed:', [n.name for n in naive_neighbors - set(cluster.neighbors.keys())])
                    offenders = list(naive_neighbors - set(cluster.neighbors.keys()))
                    self.trace_lineage(cluster, offenders[0])
                elif set(cluster.neighbors.keys()) - naive_neighbors:
                    print(depth, cluster.name, ':', [n.name for n in naive_neighbors])
                    print(depth, cluster.name, 'extra:', [n.name for n in set(cluster.neighbors.keys()) - naive_neighbors])
                    offenders = list(naive_neighbors - set(cluster.neighbors.keys()))
                    self.trace_lineage(cluster, offenders[0])
                # self.assertTrue(len(neighbors) >= len(set(cluster.neighbors.keys())))
                self.assertSetEqual(naive_neighbors, set(cluster.neighbors.keys()))
