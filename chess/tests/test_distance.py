import unittest

from chess.distance import *


class TestDistanceFunctions(unittest.TestCase):
    def test_array_attributes(self):
        x = [1, 1]
        self.assertRaises(TypeError, check_input_array, x)

        x = np.asarray([1, 1])
        self.assertRaises(ValueError, check_input_array, x)

        x = np.asarray([])
        self.assertRaises(ValueError, check_input_array, x)

        x = np.asarray([[]])
        self.assertRaises(ValueError, check_input_array, x)

        x = np.asarray([[1, 1], [2, 2], [3, 3]])
        self.assertTrue(check_input_array(x))

    def test_distance_values(self):
        x = np.asarray([[1, 1], [2, 2], [3, 3]])
        y = np.asarray([[1, 1], [2, 2], [3, 3]])

        metric = 'euclidean'
        correct_distances = np.asarray([[0, np.sqrt(2), np.sqrt(8)],
                                        [np.sqrt(2), 0, np.sqrt(2)],
                                        [np.sqrt(8), np.sqrt(2), 0]])
        distances = calculate_distances(x, y, metric)
        np.testing.assert_almost_equal(distances, correct_distances, decimal=7)

        metric = 'cosine'
        distances = calculate_distances(x, y, metric)
        np.testing.assert_almost_equal(distances, np.zeros_like(distances), decimal=7)

        metric = 'hamming'
        correct_distances = np.asarray([[0, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 0]])
        distances = calculate_distances(x, y, metric)
        np.testing.assert_almost_equal(distances, correct_distances, decimal=0)

    def test_distance_shapes(self):
        x = np.asarray([[1, 1]])
        y = np.asarray([[1, 1], [2, 2], [3, 3]])
        metric = 'euclidean'

        distances = calculate_distances(x, y, metric)[0, :]
        self.assertEqual(distances.ndim, 1)
        self.assertEqual(distances.shape, (y.shape[0],))

        distances = calculate_distances(y, x, metric)[:, 0]
        self.assertEqual(distances.ndim, 1)
        self.assertEqual(distances.shape, (y.shape[0],))
