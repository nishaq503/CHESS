import unittest

from chess.criterion import *
from chess.manifold import *


class TestCriterion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.random.randn(100, 2)
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

    @unittest.skip
    def test_leaves_component(self):
        self.assertEqual(1, len(self.manifold.graphs[-1].components))
        self.manifold.build(LeavesComponent(self.manifold))
        self.assertGreater(len(self.manifold.graphs[-1].components), 1)
        self.assertLess(len(self.manifold.graphs[-1].components), self.data.shape[0])
        return

    def test_min_cardinality(self):
        self.manifold.build(MinCardinality(1))
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
