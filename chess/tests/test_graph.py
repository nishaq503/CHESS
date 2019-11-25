import unittest

from chess import CHESS
from chess.graph import *


class TestGraph(unittest.TestCase):
    tempfile: str

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(42)
        data = np.random.randn(1_000, 100)
        cls.data = np.concatenate([data - 10_000, data + 10_000])
        # noinspection PyTypeChecker
        cls.chess_object = CHESS(data=cls.data, metric='euclidean', max_depth=10, min_points=10, min_radius=0.5)
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

    # TODO: Think of better tests that confirm the components are actually disconnected.
    def test_connected_clusters(self):
        num_components = [1]
        for d in range(1, self.max_depth + 1):
            g = graph(list(self.chess_object.root.leaves(d)))
            components = connected_clusters(g)
            self.assertTrue(num_components[-1] <= len(components))
            num_components.append(len(components))
            if d == self.max_depth:
                self.assertLessEqual(len(components), self.data.shape[0])
        self.assertTrue(len(num_components) > 1)
        self.assertTrue(any([nc == 2 for nc in num_components]))
        self.assertTrue(num_components[-1] > num_components[0])
        return

    # TODO: Think of better tests that confirm the subgraphs are actually disconnected.
    def test_subgraphs(self):
        num_components = [1]
        for d in range(1, self.max_depth + 1):
            g = graph(list(self.chess_object.root.leaves(d)))
            components = subgraphs(g)
            self.assertTrue(num_components[-1] <= len(components))
            if d > 1:
                num_components.append(len(components))
            if d == self.max_depth:
                self.assertLessEqual(len(components), self.data.shape[0])
        self.assertTrue(len(num_components) > 1)
        self.assertTrue(any([nc == 2 for nc in num_components]))
        self.assertTrue(num_components[-1] > num_components[0])
        return
