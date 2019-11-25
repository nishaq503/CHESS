import unittest

from chess import CHESS
from chess.graph import *


class TestGraph(unittest.TestCase):
    tempfile: str

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(42)
        data = np.random.randn(100, 100)
        cls.data = np.concatenate([data - 10_000, data + 10_000])
        cls.chess_object = CHESS(cls.data, 'euclidean')
        cls.chess_object.build()
        cls.max_depth = max(map(len, cls.chess_object.root.dict().keys()))
        return

    @classmethod
    def tearDownClass(cls) -> None:
        return

    def test_graph_building(self):
        for d in range(1, self.max_depth + 1):
            leaves: List[Cluster] = list(self.chess_object.root.leaves(d))
            g = graph(leaves)
            self.assertSetEqual(set([l for l in leaves]), set(g.keys()))
            self.assertEqual(len(leaves), len(list(g.keys())))

    def test_connected_clusters(self):
        num_components = [1]
        for d in range(1, self.max_depth + 1):
            g = graph(list(self.chess_object.root.leaves(d)))
            components = connected_clusters(g)
            if d > 1:
                num_components.append(len(components))
            self.assertTrue(num_components[d - 1] <= len(components))
            if d == self.max_depth:
                self.assertLessEqual(len(components), self.data.shape[0])
        return

    def test_subgraphs(self):
        num_components = [1]
        for d in range(1, self.max_depth + 1):
            g = graph(list(self.chess_object.root.leaves(d)))
            _components = subgraphs(g)
            if d > 1:
                num_components.append(len(_components))
            self.assertTrue(num_components[d - 1] <= len(_components))
            if d == self.max_depth:
                self.assertLessEqual(len(_components), self.data.shape[0])
        return
