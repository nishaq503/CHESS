import unittest

from chess.datasets import *


class TestDatasets(unittest.TestCase):

    def test_load(self):
        d = load('Apogee')
        self.assertIs(d, Apogee)
        with self.assertRaises(ValueError):
            load('noDatasetShouldEverBeNamedThisForSure')
