""" Query Object.

This helper-class wraps up a numpy array along with some combination
of optional parameters to generate a valid search object.
"""
import numpy as np

from chess import defaults


class Query:
    """ Query object. """

    def __init__(
            self,
            point: np.ndarray,
            *,
            radius: defaults.RADII_DTYPE = 0,
            k: int = 0,
            max_depth: int = np.inf
    ):
        self.point = point
        self.radius = radius
        self.k = k
        self.max_depth = max_depth
