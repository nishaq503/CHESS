""" Global configuration.
"""
import numpy as np

H_MAGNITUDE = 12.2

BATCH_SIZE = 10_000
SUBSAMPLING_LIMIT = 100
MIN_POINTS = 10
MIN_RADIUS = 0
MAX_DEPTH = 50
RADII_DTYPE = np.float64

SEARCH_RADII = {
    'euclidean': [2000, 4000],
    'cosine': [0.005, 0.001],
    'hamming': [0.001, 0.01],
}
DF_CALLS = 0
