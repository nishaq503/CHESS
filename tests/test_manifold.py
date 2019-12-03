import unittest

from chess.manifold import *


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
