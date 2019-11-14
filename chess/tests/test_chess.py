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
        cluster = CHESS(self.data, 'euclidean')
