""" Preconfigured Datasets.

CHESS comes shipped with configurations for Apogee and GreenGenes data,
both of which should correspond directly to the initial publication of
the CHESS paper.
"""
import logging
from abc import ABC
from inspect import isclass

import numpy as np

log = logging.getLogger(__name__)


class Dataset(ABC):
    FULL: str
    DATA: str
    QUERIES: str
    NQUERIES: int
    DTYPE: str
    ROWS: int
    DIMS: int

    def get_data(self):
        return np.memmap(
            filename=self.DATA,
            dtype=self.DTYPE,
            mode='r',
            shape=self.data_shape(),
        )

    def data_shape(self):
        return self.ROWS - self.NQUERIES, self.DIMS

    def get_queries(self):
        return np.memmap(
            filename=self.QUERIES,
            dtype=self.DTYPE,
            mode='r',
            shape=self.queries_shape(),
        )

    def queries_shape(self):
        return self.NQUERIES, self.DIMS


class Apogee(Dataset):
    FULL = '/home/nishaq/APOGEE/apogee_full.memmap'
    DATA = '/home/nishaq/APOGEE/apogee_data.memmap'
    QUERIES = '/home/nishaq/APOGEE/apogee_queries.memmap'
    NUM_ROWS = 134_510 - 810
    NUM_QUERIES = 10_000
    NUM_DIMS = 8_575
    DATA_SHAPE = (NUM_ROWS - NUM_QUERIES, NUM_DIMS)
    QUERIES_SHAPE = (NUM_QUERIES, NUM_DIMS)
    DTYPE = 'float32'


class GreenGenes(Dataset):
    FULL = '/home/nishaq/GreenGenes/gg_full.memmap'
    DATA = '/home/nishaq/GreenGenes/gg_data.memmap'
    QUERIES = '/home/nishaq/GreenGenes/gg_queries.memmap'
    DATA_NO_DUP = '/home/nishaq/GreenGenes/gg_data_no_dup.memmap'
    QUERIES_NO_DUP = '/home/nishaq/GreenGenes/gg_queries_no_dup.memmap'
    NUM_ROWS = 1_027_383
    NUM_QUERIES = 10_000
    NUM_DATA_NO_DUP = 805_434
    NUM_QUERIES_NO_DUP = 9_470
    NUM_DIMS = 7_682
    DATA_SHAPE_NO_DUP = (NUM_DATA_NO_DUP, NUM_DIMS)
    QUERIES_SHAPE_NO_DUP = (NUM_QUERIES_NO_DUP, NUM_DIMS)
    DATA_SHAPE = (NUM_ROWS - NUM_QUERIES, NUM_DIMS)
    QUERIES_SHAPE = (NUM_QUERIES, NUM_DIMS)
    DTYPE = np.int8


# Gather all subclasses of Dataset.
# noinspection PyTypeChecker
DATASETS = dict(filter(
    lambda e: isclass(e[1]) and e[1] is not Dataset and issubclass(e[1], Dataset),
    globals().items()
))


def load(dataset: str) -> Dataset:
    """ Attempts to load the given dataset.

    :param dataset: the name of the dataset to load.

    :return Dataset: the selected dataset.
    """
    try:
        return DATASETS[dataset]
    except KeyError:
        raise ValueError(f'Invalid dataset selected. Choices are: {DATASETS}')


def filter_duplicates(data: np.memmap, filename: str):
    # TODO: Switch to np.unique()
    set_data = set()
    [set_data.add(tuple(point)) for point in data]

    log.debug(f'Reduced from {data.shape}, to {len(set_data)}')
    if data.shape[0] == len(set_data):
        # No elements were removed.
        return

    def write_memmap(set_to_write):
        my_memmap = np.memmap(
            filename=filename,
            dtype=data.dtype,
            mode='w+',
            shape=(len(set_to_write), *data.shape[1:]),
        )
        for i, point in enumerate(set_to_write):
            p = np.asarray(point, dtype=data.dtype)
            my_memmap[i] = p
        my_memmap.flush()

    write_memmap(set_data)
    return
