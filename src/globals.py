import numpy as np

APOGEE_FULL = '/home/nishaq/APOGEE/apogee_full.memmap'
APOGEE_DATA = '/home/nishaq/APOGEE/apogee_data.memmap'
APOGEE_QUERIES = '/home/nishaq/APOGEE/apogee_queries.memmap'
APOGEE_NUM_ROWS = 134_510 - 810
APOGEE_NUM_QUERIES = 10_000
APOGEE_NUM_DIMS = 8_575
APOGEE_DATA_SHAPE = (APOGEE_NUM_ROWS - APOGEE_NUM_QUERIES, APOGEE_NUM_DIMS)
APOGEE_QUERIES_SHAPE = (APOGEE_NUM_QUERIES, APOGEE_NUM_DIMS)
APOGEE_DTYPE = 'float32'
H_MAGNITUDE = 12.2

GREENGENES_FULL = '/home/nishaq/GreenGenes/gg_full.memmap'
GREENGENES_DATA = '/home/nishaq/GreenGenes/gg_data.memmap'
GREENGENES_QUERIES = '/home/nishaq/GreenGenes/gg_queries.memmap'
GREENGENES_NUM_ROWS = 1_027_383
GREENGENES_NUM_QUERIES = 10_000
GREENGENES_NUM_DIMS = 7_682
GREENGENES_DATA_SHAPE = (GREENGENES_NUM_ROWS - GREENGENES_NUM_QUERIES, GREENGENES_NUM_DIMS)
GREENGENES_QUERIES_SHAPE = (GREENGENES_NUM_QUERIES, GREENGENES_NUM_DIMS)
GREENGENES_DTYPE = np.int8

BATCH_SIZE = 10_000
SUBSAMPLING_LIMIT = 100
MIN_POINTS = 10
MIN_RADIUS = 0
MAX_DEPTH = 50
RADII_DTYPE = np.float64

DISTANCE_FUNCTIONS = {
    'euclidean',
    'cosine',
    'hamming',
}
DF_CALLS = 0
