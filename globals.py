import numpy as np

APOGEE_FULL = '/data/nishaq/APOGEE/apogee_full.memmap'
APOGEE_DATA = '/data/nishaq/APOGEE/apogee_data.memmap'
APOGEE_QUERIES = '/data/nishaq/APOGEE/apogee_queries.memmap'
APOGEE_NUM_ROWS = 134_510 - 810
APOGEE_NUM_QUERIES = 10_000
APOGEE_NUM_DIMS = 8_575
APOGEE_DTYPE = 'float32'
H_MAGNITUDE = 12.2

GREENGENES_FULL = '/data/nishaq/GreenGenes/gg_full.memmap'
GREENGENES_DATA = '/data/nishaq/GreenGenes/gg_data.memmap'
GREENGENES_QUERIES = '/data/nishaq/GreenGenes/gg_queries.memmap'
GREENGENES_NUM_ROWS = 1_027_383
GREENGENES_NUM_QUERIES = 10_000
GREENGENES_NUM_DIMS = 7_682
GREENGENES_DTYPE = np.int8

BATCH_SIZE = 10_000
TF_VS_NP_NUM = 100
MIN_POINTS = 10
MIN_RADIUS = 0
MAX_DEPTH = 50

DF_CALLS = 0
