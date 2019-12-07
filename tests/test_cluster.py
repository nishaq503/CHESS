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

    def test_points(self):  # TODO: Tom
        # self.assertTrue(np.array_equal(
        #     self.manifold.data,
        #     self.cluster.points  # iterator needs to be made list
        # ))
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
    def trace_lineage(left: Cluster, right: Cluster):  # TODO: Cover
        assert left.depth == right.depth
        assert left.overlaps(right.center, right.radius)
        lineages = [(left.name[:i], right.name[:i]) for i in range(left.depth) if left.name[:i] != right.name[:i]]
        ancestors = [(left.manifold.select(l_), right.manifold.select(r_)) for l_, r_ in reversed(lineages)]
        for al, ar in ancestors:
            print(f'checking {al.name, ar.name}...')
            if not al.overlaps(ar.center, ar.radius):
                print(f'{al.name, ar.name} do not have overlap but their descendents {left.name, right.name} do.')
                # noinspection PyTypeChecker
                d_ancestors, r_ancestors = al.distance([ar.center])[0], al.radius + ar.radius
                # noinspection PyTypeChecker
                d, r = left.distance([right.center])[0], left.radius + right.radius
                print(f'ancestors\' distance: {d_ancestors}, radii_sum: {r_ancestors}')
                print(f'children\'s distance: {d}, radii_sum: {r}')
                return
        else:
            raise ValueError(f'all divergent ancestors had overlap')

    def test_neighbors_more(self):
        data, labels = spiral_2d()
        np.random.seed(42)
        manifold = Manifold(data, 'euclidean', propagate=True)
        manifold.build(MinRadius(MIN_RADIUS), MaxDepth(10))
        for depth, graph in enumerate(manifold.graphs):
            for cluster in graph:
                potential_neighbors = [c for c in graph if c.name != cluster.name]
                if len(potential_neighbors) == 0:
                    continue
                elif len(potential_neighbors) == 1:
                    centers = np.expand_dims(potential_neighbors[0].center, axis=0)
                else:
                    centers = np.stack([c.center for c in potential_neighbors])
                distances = list(cluster.distance(centers))
                radii = [cluster.radius + c.radius for c in potential_neighbors]  # TODO: Slow
                naive_neighbors = {c for c, d, r in zip(potential_neighbors, distances, radii) if d <= r}
                if naive_neighbors - set(cluster.neighbors.keys()):  # TODO: Cover
                    offenders = list(naive_neighbors - set(cluster.neighbors.keys()))
                    [self.trace_lineage(cluster, o) for o in offenders]
                self.assertSetEqual(set(), (naive_neighbors - set(cluster.neighbors.keys())),
                                    msg=f'\nmissed: {sorted([n.name for n in naive_neighbors - set(cluster.neighbors.keys())])}'
                                        f'\nextras: {sorted([n.name for n in set(cluster.neighbors.keys()) - naive_neighbors])}')

    def test_add_edge_with_propagation(self):
        data = np.random.random((1000, 2))
        data = np.concatenate([data - 100, data + 100], axis=0)
        manifold = Manifold(data, 'euclidean')
        manifold.build(MaxDepth(5), MinPoints(1))
        bottom = manifold.graphs[-1]
        for _ in range(100):
            c1: Cluster
            c2: Cluster
            c1 = next(iter(bottom))
            for c2 in bottom:
                if c1.name != c2.name and c1 not in set(c2.neighbors.keys()):
                    break
            else:
                continue
            c1.add_edge(c2, True)

            lineages = [(c1.name[:i], c2.name[:i]) for i in range(c1.depth) if c1.name[:i] != c2.name[:i]]
            clusters = [(manifold.select(l1), manifold.select(l2)) for l1, l2 in lineages]
            for l1, l2 in reversed(clusters):
                self.assertNotEqual(l1.name, l2.name)
                self.assertIn(l1, set(l2.neighbors.keys()), msg=f'{l2.name} did not have an edge with {l1.name} but {c1.name} and {c2.name} have an edge.')
                self.assertIn(l2, set(l1.neighbors.keys()), msg=f'{l1.name} did not have an edge with {l2.name} but {c1.name} and {c2.name} have an edge.')
