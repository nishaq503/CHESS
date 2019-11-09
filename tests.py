import unittest
from typing import Set

import numpy as np

from src.distance_functions import check_input_array, calculate_distances
from src.search import get_data_and_queries
from src import globals


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


def sandbox():
    data, queries = get_data_and_queries(dataset='GreenGenes')
    set_data: Set = set()
    [set_data.add(tuple(point)) for point in queries]
    print(len(queries), len(set_data))
    return


def filter_duplicates():
    data, queries = get_data_and_queries(dataset='GreenGenes')
    set_data, set_queries = set(), set()
    [set_data.add(tuple(point)) for point in data]
    [set_queries.add(tuple(point)) for point in queries]

    print(len(set_data), len(set_queries))
    with open('lengths.txt', 'a') as outfile:
        outfile.write(f'GREENGENES_NUM_DATA_NO_DUP = {len(set_data)}\n')
        outfile.write(f'GREENGENES_NUM_QUERIES_NO_DUP = {len(set_queries)}\n')

    def write_memmap(filename, set_to_write):
        my_memmap = np.memmap(
            filename=filename,
            dtype=globals.GREENGENES_DTYPE,
            mode='w+',
            shape=(len(set_to_write), globals.GREENGENES_NUM_DIMS),
        )
        for i, point in enumerate(set_to_write):
            p = np.asarray(point, dtype=globals.GREENGENES_DTYPE)
            my_memmap[i] = p
        my_memmap.flush()
        del my_memmap

    write_memmap(globals.GREENGENES_DATA_NO_DUP, set_data)
    write_memmap(globals.GREENGENES_QUERIES_NO_DUP, set_queries)

    return


if __name__ == '__main__':
    # unittest.main()
    sandbox()
