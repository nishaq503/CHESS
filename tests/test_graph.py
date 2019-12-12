import unittest

from chess.criterion import *
from chess.manifold import *


class TestGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.random.randn(100, 10)
        cls.manifold = Manifold(cls.data, 'euclidean')
        cls.manifold.build(MaxDepth(5))
        return

    def test_init(self):
        raise NotImplementedError

    def test_eq(self):
        raise NotImplementedError

    def test_iter(self):
        raise NotImplementedError

    def test_len(self):
        raise NotImplementedError

    def test_str(self):
        raise NotImplementedError

    def test_repr(self):
        raise NotImplementedError

    def test_contains(self):
        raise NotImplementedError

    def test_manifold(self):
        raise NotImplementedError

    def test_depth(self):
        raise NotImplementedError

    def test_edges(self):
        raise NotImplementedError

    def test_subgraphs(self):
        raise NotImplementedError

    def test_components(self):
        raise NotImplementedError

    def test_clear_cache(self):
        raise NotImplementedError

    def test_component(self):
        raise NotImplementedError

    def test_bft(self):
        raise NotImplementedError

    def test_dft(self):
        raise NotImplementedError
