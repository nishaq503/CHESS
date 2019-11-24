import tempfile
import unittest

from chess import CHESS
from chess.graph import *


class TestGraph(unittest.TestCase):
    tempfile: str

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(42)
        cls.tempfile = tempfile.NamedTemporaryFile()

        scale, num_points = 12, 10_000
        samples = scale * (np.random.rand(2, num_points) - 0.5)
        distances = np.linalg.norm(samples, axis=0)
        x = [samples[0, i] for i in range(num_points)
             if distances[i] < 2 or (4 < distances[i] < 6)]
        y = [samples[0, i] for i in range(num_points)
             if distances[i] < 2 or (4 < distances[i] < 6)]

        data = np.asarray((x, y), dtype=defaults.RADII_DTYPE).T
        cls.data = np.memmap(cls.tempfile, dtype='float32', mode='w+', shape=data.shape)
        cls.data[:] = data[:]

        cls.chess_object: CHESS = CHESS(
            data=data,
            metric='euclidean',
            max_depth=25,
            min_points=1,
            min_radius=defaults.RADII_DTYPE(0)
        )
        cls.chess_object.build()
        cls.max_depth = max(map(len, cls.chess_object.root.dict().keys()))
        return

    @classmethod
    def tearDownClass(cls) -> None:
        return

    def test_graph_building(self):
        np.random.seed(42)
        for d in range(1, self.max_depth + 1):
            leaves: List[Cluster] = list(self.chess_object.root.leaves(d))
            g = graph(leaves)
            self.assertSetEqual(set([l for l in leaves]), set(g.keys()))
            self.assertEqual(len(leaves), len(list(g.keys())))

    def test_connected_components(self):
        np.random.seed(42)
        num_components = [1]
        for d in range(1, self.max_depth + 1):
            leaves: List[Cluster] = list(self.chess_object.root.leaves(d))
            g = graph(leaves)
            components = connected_components(g)
            if d > 1:
                num_components.append(len(components))
            self.assertTrue(num_components[d - 1] <= len(components))
            if d == self.max_depth:
                self.assertEqual(len(components), self.data.shape[0])
        return
