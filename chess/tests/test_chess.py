import os
import tempfile
import unittest

import numpy as np

from chess.chess import CHESS


class TestCHESS(unittest.TestCase):
    tempfile: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempfile = tempfile.NamedTemporaryFile()
        data = np.random.randn(100, 100)
        cls.data = np.memmap(cls.tempfile, dtype='float32', mode='w+', shape=data.shape)
        cls.data[:] = data[:]
        return

    @classmethod
    def tearDownClass(cls) -> None:
        return

    def test_functional(self):
        chess = CHESS(self.data, 'euclidean')
        chess.build()
        result = chess.search(self.data[0], 0.0)
        self.assertEqual(len(result), 1)

        result = chess.search(self.data[0] + 50, 0.0)
        self.assertEqual(len(result), 0)

        result = chess.search(self.data[0], 20.0)
        self.assertEqual(len(result), 100)
        return

    def test_init(self):
        CHESS(self.data, 'euclidean')

        with self.assertRaises(ValueError):
            CHESS(self.data, 'boopscooparoop')

        return

    def test_str(self):
        chess = CHESS(self.data, 'euclidean')
        chess.build()
        s = str(chess)
        self.assertGreater(len(s), 0)
        # Do we have the right number of clusters? (Exclude title row)
        self.assertEqual(len(s.split('\n')[1:]), len([c for c in chess.cluster.leaves()]))
        points = [p for s in s.split('\n')[1:] for p in eval(s[s.index('[') - 1:])]
        self.assertEqual(len(points), 100)
        return

    def test_repr(self):
        chess = CHESS(self.data, 'euclidean')
        chess.build()
        s = repr(chess)
        self.assertGreater(len(s), 0)
        return

    def test_build(self):
        chess = CHESS(self.data, 'euclidean')
        chess.build()
        self.assertTrue(chess.cluster.left and chess.cluster.right)
        self.assertTrue(
            (chess.cluster.left.left and chess.cluster.left.right)
            or (chess.cluster.right.left and chess.cluster.right.right)
        )
        return

    def test_search(self):
        chess = CHESS(self.data, 'euclidean')
        chess.build()
        self.assertEqual(len(chess.search(self.data[0], 0.0)), 1)
        self.assertEqual(len(chess.search(self.data[0] + 100, 0.0)), 0)
        self.assertGreaterEqual(len(chess.search(self.data[0] + 0.1, 10.0)), 1)
        self.assertEqual(len(chess.search(self.data[0], 100)), 100)
        return

    def test_knn_search(self):
        pass

    def test_compress(self):
        chess = CHESS(self.data, 'euclidean')
        chess.build()
        filepath = tempfile.NamedTemporaryFile()
        chess.compress(filepath)
        data = np.memmap(filepath, mode='r+', dtype='float32', shape=self.data.shape)
        self.assertEqual(data.shape, (100, 100))
        filepath.close()
        return

    def test_write(self):
        chess = CHESS(self.data, 'euclidean')
        with tempfile.TemporaryDirectory() as d:
            chess.write(os.path.join(d, 'dump'))
            self.assertTrue(os.path.exists(os.path.join(d, 'dump')))

    def test_load(self):
        chess = CHESS(self.data, 'euclidean')
        with tempfile.TemporaryDirectory() as d:
            chess.write(os.path.join(d, 'dump'))
            loaded = CHESS.load(os.path.join(d, 'dump'))
        self.assertEqual(chess, loaded)
