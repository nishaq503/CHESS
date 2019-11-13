import logging

import numpy as np

log = logging.getLogger(__name__)


def get_data(dataset: str, mode: str = 'r') -> np.memmap:
    """ Reads and returns the numpy memmap file for the given dataset.

    :param dataset: data set to read. Must be APOGEE or GreenGenes.
    :param mode: optional mode to read the memmap files in.
    :return: data for clustering.
    """
    if dataset == 'APOGEE':
        data = np.memmap(
            filename=Apogee.DATA,
            dtype=Apogee.DTYPE,
            mode=mode,
            shape=Apogee.DATA_SHAPE,
        )
    elif dataset == 'GreenGenes':
        data = np.memmap(
            filename=GreenGenes.DATA,
            dtype=GreenGenes.DTYPE,
            mode=mode,
            shape=GreenGenes.DATA_SHAPE,
        )
    else:
        raise ValueError(f'Only the APOGEE and GreenGenes datasets are available. Got {dataset}.')

    return data


def filter_duplicates(data: np.memmap, filename: str):
    # TODO: Switch to np.unique()
    set_data = set()
    [set_data.add(tuple(point)) for point in data]

    log.debug(f'Reduced from {data.shape}, to {len(set_data)}')
    if data.shape[0] == len(set_data):
        # No elements were removed.
        return

    def write_memmap(filename, set_to_write):
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
        del my_memmap

    write_memmap(filename, set_data)
    return


class Apogee:
    FULL = '/home/nishaq/APOGEE/apogee_full.memmap'
    DATA = '/home/nishaq/APOGEE/apogee_data.memmap'
    QUERIES = '/home/nishaq/APOGEE/apogee_queries.memmap'
    NUM_ROWS = 134_510 - 810
    NUM_QUERIES = 10_000
    NUM_DIMS = 8_575
    DATA_SHAPE = (NUM_ROWS - NUM_QUERIES, NUM_DIMS)
    QUERIES_SHAPE = (NUM_QUERIES, NUM_DIMS)
    DTYPE = 'float32'


class GreenGenes:
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
