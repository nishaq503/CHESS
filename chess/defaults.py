""" Default configuration.
"""
from typing import Callable, Union

import numpy as np

H_MAGNITUDE: float = 12.2

RADII_DTYPE = np.float64
BATCH_SIZE: int = 10_000
SUBSAMPLING_LIMIT: int = 100
MIN_POINTS: int = 10
MIN_RADIUS: RADII_DTYPE = RADII_DTYPE(0)
MAX_DEPTH: int = 50
STOPPING_CRITERIA: Union[Callable[[any], bool], None] = None

SEARCH_RADII = {
    'euclidean': [2000, 4000],
    'cosine': [0.005, 0.001],
    'hamming': [0.001, 0.01],
}
DF_CALLS: int = 0
