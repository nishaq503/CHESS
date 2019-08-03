from typing import List

import numpy as np

import config


class Cluster:
    """ Contains information relevant to a cluster. Defines the relevant methods as well. """

    def __init__(
            self,
            data: np.memmap,
            points: List[int],
            distance_function: str,
            name: str,
            center: int = None,
            radius: float = None,
            lfd: float = None,
            parent_lfd: float = 0.0,
            left=None,
            right=None,
            reading: bool = False,
    ):
        """ Initializes cluster.

        :param data:
        :param points:
        :param distance_function:
        :param name:
        :param center:
        :param radius:
        :param lfd:
        :param parent_lfd:
        :param left:
        :param right:
        :param reading:
        """

        self.name: str = name
        self.depth: int = len(self.name)

        self.data: np.memmap = data
        self.points: List[int] = points
        self.distance_function: str = distance_function
        self.parent_lfd: float = parent_lfd
        self.left: Cluster = left
        self.right: Cluster = right

        self.num_dims = config.NUM_DIMS
        self.batch_size = config.BATCH_SIZE
        self.max_depth = config.MAX_DEPTH
        self.min_points = config.MIN_POINTS
        self.min_radius = config.MIN_RADIUS
        self.should_subsample: bool = len(points) > config.BATCH_SIZE

        self._potential_centers = None
        self._pairwise_distances = None

        if reading:
            self.center: int = center
            self.radius: float = radius
            self.lfd: float = lfd
        else:
            self.update()

    def update(self):
        pass
