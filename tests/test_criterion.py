import unittest

import numpy as np

from chess.criterion import *


class TestCriterion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.random.randn(100, 2)
        return

    def setUp(self) -> None:
        self.manifold = Manifold(self.data, 'euclidean')
        return

    def test_min_radius(self):
        min_radius = 1.
        self.manifold.build(MinRadius(min_radius))
        self.assertTrue(all((c.radius >= min_radius for g in self.manifold for c in g)))
