import tempfile
import unittest
from typing import List

import numpy as np
from chess import CHESS, defaults
from chess.cluster import Cluster
from chess.graph import Graph


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
             if distances[i] < 6 and (distances[i] > 4 or distances[i] < 2)]
        y = [samples[1, i] for i in range(num_points)
             if distances[i] < 6 and (distances[i] > 4 or distances[i] < 2)]

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

    def test_init(self):
        np.random.seed(42)
        for d in range(1, self.max_depth + 1):
            leaves: List[Cluster] = list(self.chess_object.root.leaves(d))
            graph: Graph = Graph(
                data=self.data,
                metric='euclidean',
                leaves=leaves
            )
            self.assertEqual(self.data.shape[0], graph.data.shape[0])
            self.assertEqual(self.data.shape[1], graph.data.shape[1])
            self.assertEqual('euclidean', graph.metric)
            self.assertEqual(len(leaves), len(list(graph.graph.keys())))
            self.assertFalse(any(graph.graph.values()))
        return

    def test_connected_components(self):
        np.random.seed(42)
        num_components = [1]
        for d in range(1, self.max_depth + 1):
            leaves: List[Cluster] = list(self.chess_object.root.leaves(d))
            graph: Graph = Graph(
                data=self.data,
                metric='euclidean',
                leaves=leaves
            )
            graph.build()
            self.assertEqual(len(leaves), len(list(graph.graph.keys())))

            components = graph.connected_components()
            if d > 1:
                num_components.append(len(components))
            self.assertTrue(num_components[d - 1] <= len(components))
            if d == self.max_depth:
                self.assertEqual(len(components), self.data.shape[0])
        return
