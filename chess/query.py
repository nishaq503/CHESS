import numpy as np

from chess import globals


class Query:
    def __init__(self,
                 point: np.ndarray,
                 *,
                 radius: globals.RADII_DTYPE = 0,
                 k: int = 0,
                 max_depth: int = np.inf
                 ):
        self.point = point
        self.radius = radius
        self.k = k
        self.max_depth = max_depth
