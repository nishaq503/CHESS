import unittest
from inspect import stack
from itertools import cycle

from matplotlib import pyplot as plt

from chess.criterion import *
from chess.manifold import *


class TestCriterion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.concatenate([np.random.randn(100, 2) + i * 100 for i in range(5)])
        return

    def setUp(self) -> None:
        self.manifold = Manifold(self.data, 'euclidean')
        return

    def test_min_radius(self):
        min_radius = 0.1
        self.manifold.build(MinRadius(min_radius))
        self.assertTrue(all((c.radius >= min_radius for g in self.manifold for c in g)))
        [self.assertLessEqual(len(c.children), 1) for g in self.manifold for c in g if c.radius <= min_radius]
        return

    def test_leaves_component(self):
        self.assertEqual(1, len(self.manifold.graphs[-1].components))
        self.manifold.build(LeavesComponent(self.manifold))
        self.assertGreater(len(self.manifold.graphs[-1].components), 1)
        return

    def test_singleton_cardinality(self):
        self.assertEqual(1, len(self.manifold.graphs[-1].components))
        self.manifold.build(MinCardinality(1))
        self.assertGreater(len(self.manifold.graphs[-1].components), 1)
        self.assertTrue(all((len(c.neighbors) == 0) for c in self.manifold.graphs[-1]))
        return

    def test_min_neighborhood(self):
        self.manifold.build(MinNeighborhood(5, 1))
        return

    def test_new_component(self):
        self.manifold.build(NewComponent(self.manifold))
        return

    def test_combinations(self):
        min_radius, min_points, max_depth = 0.15, 10, 20
        self.manifold.build(MinRadius(min_radius), MinPoints(min_points), MaxDepth(max_depth))
        self.assertTrue(all((c.radius >= min_radius for g in self.manifold for c in g)))
        [self.assertLessEqual(len(c.children), 1) for g in self.manifold.graphs for c in g
         if c.radius <= min_radius or len(c.argpoints) <= min_points or c.depth >= max_depth]
        self.plot()

    def test_medoid_near_centroid(self):
        self.manifold.build(MedoidNearCentroid())
        self.plot()
        return

    def plot(self):
        colors = cycle('bgrcmy')
        shapes = cycle('.ov^<>1234spP*+xXD|')
        for graph in self.manifold.graphs:
            fig, ax = plt.subplots()
            for cluster in graph:
                marker = next(colors) + next(shapes)

                # The points themselves.
                [plt.plot(batch[:, 0], batch[:, 1], marker, alpha=0.7) for batch in cluster.data]
                # plt.plot(*[d for batch in cluster.data for d in (batch[:, 0], batch[:, 1], marker)])

                # Centroid
                c = plt.Circle(tuple(cluster.medoid), cluster.radius, fill=False, color='k', zorder=10)
                ax.add_artist(c)
            plt.title(stack()[1].function)
            plt.show()
