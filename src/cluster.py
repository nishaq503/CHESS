import pickle
from typing import List

import numpy as np

import config
from src.utils import tf_calculate_pairwise_distances, numpy_calculate_distance, tf_calculate_distance


class Cluster:
    """ Defines clusters and methods relevant to building a cluster-tree and for searching the tree. """

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
        """ Initializes Cluster object.

        :param data: memmap of points to cluster.
        :param points: list of indexes of points in the cluster.
        :param distance_function: distance function to use for the cluster.
        :param name: name of cluster.
        :param center: index of center of cluster.
        :param radius: radius of cluster.
        :param lfd: local fractal dimension of cluster.
        :param parent_lfd: local fractal dimension of parent cluster.
        :param left: left child cluster.
        :param right: right child cluster.
        :param reading: weather of not a cluster-tree is being read from file.
        """
        self.name: str = name
        self.depth: int = len(self.name)

        self.data: np.memmap = data
        self.points: List[int] = points
        self.df: str = distance_function
        self.parent_lfd: float = parent_lfd
        self.left: Cluster = left
        self.right: Cluster = right

        self.batch_size = config.BATCH_SIZE
        self.max_depth = max(config.MAX_DEPTH, len(name))
        self.min_points = config.MIN_POINTS
        self.min_radius = config.MIN_RADIUS
        self.should_subsample: bool = len(points) > config.NP_PTS

        self._potential_centers = None
        self._pairwise_distances = None

        if reading:
            self.center: int = center
            self.radius: float = radius
            self.lfd: float = lfd
        else:
            self.update()

    def _get_potential_centers(self) -> List[int]:
        """ If the cluster contains too many points, subsample square_root as many points as potential centers.

        :return: list of indexes of potential centers.
        """
        if self.should_subsample:
            sample_size = int(np.sqrt(len(self.points)))
            points = self.points.copy()
            np.random.shuffle(points)
            return points[: sample_size + 1]
        else:
            return self.points.copy()

    def _calculate_pairwise_distances(self) -> np.ndarray:
        """ Calculate the pairwise distances between potential centers.

        :return: pairwise distance matrix between potential centers.
        """
        if self.should_subsample:
            points = np.asarray([self.data[p] for p in self._potential_centers])
            return tf_calculate_pairwise_distances(points, self.df)
        else:
            points = np.asarray([self.data[p] for p in self.points])
            return numpy_calculate_distance(points, points, self.df)

    def _calculate_center(self) -> int:
        """ Calculate the geometric median of the potential centers. This is the center of the cluster.

        :return: index of center of cluster.
        """
        sum_distances = np.sum(self._pairwise_distances, axis=1)
        return self._potential_centers[int(np.argmin(sum_distances))]

    def _get_batch(self, start):
        num_points = min(start + self.batch_size, len(self.points)) - start
        return np.asarray([self.data[p] for p in self.points[start: start + num_points]])

    def _calculate_radius(self) -> float:
        """ Calculate the radius of the cluster.

        :return: cluster radius.
        """
        if self.should_subsample:
            return max(max([max(tf_calculate_distance(self.data[self.center], self._get_batch(i), self.df))
                            for i in range(0, len(self.points), self.batch_size)]), 0.0)
        else:
            points = np.asarray([self.data[p] for p in self.points])
            distances = numpy_calculate_distance(self.data[self.center], points, self.df)
            return max(max(distances), 0.0)

    def _calculate_lfd(self) -> float:
        """ Calculate the local fractal dimension of the cluster.

        :return: local fractal dimension of cluster.
        """
        if self.should_subsample:
            count = sum([sum([1 if d < self.radius / 2 else 0
                              for d in tf_calculate_distance(self.data[self.center],
                                                             self._get_batch(i),
                                                             self.df)])
                         for i in range(0, len(self.points), self.batch_size)])
        else:
            points = np.asarray([self.data[p] for p in self.points])
            distances = numpy_calculate_distance(self.data[self.center], points, self.df)
            count = sum([1 if d < self.radius / 2 else 0 for d in distances])

        return 0 if count == 0 else np.log2(len(self.points) / count)

    def update(self, internals_only: bool = False) -> None:
        """ Updates the relevant private variables of the cluster. """
        self._potential_centers: List[int] = self._get_potential_centers()
        self._pairwise_distances: np.ndarray = self._calculate_pairwise_distances()

        if not internals_only:
            self.center: int = self._calculate_center()
            self.radius: float = self._calculate_radius()
            self.lfd: float = self._calculate_lfd()
        return

    def can_include(self, query: np.ndarray, radius: float = 0) -> bool:
        """ Weather or not the query can be included in the cluster.

        :param query: point to check.
        :param radius: if the point is a search query, add the radius to cluster_radius.
        :return:
        """
        distance = numpy_calculate_distance(query, self.data[self.center], self.df)
        return distance <= (self.radius + radius)

    def can_be_popped(self) -> bool:
        """ check if the cluster can be popped.

        :return: Weather or not the cluster can be popped.
        """
        return all((
            len(self.points) > self.min_points,
            self.radius > self.min_radius,
            self.depth < self.max_depth,
            self.lfd < config.LFD_LIMIT or self.lfd > self.parent_lfd,
        ))

    def pop(self) -> None:
        """ Pop the cluster into left and right children.

        Steps:
            * Check if cluster has already been popped or if it can be popped.
            * Find the two potential centers that are the farthest apart and treat them as the left and right poles.
            * Partition the points in the cluster by which pole they are closer to.
            * assign the partitioned lists of points to left or right child cluster appropriately.

        :return:
        """
        if self.left or self.right or (not self.can_be_popped()):
            return

        max_pair_index = np.argmax(self._pairwise_distances)
        max_col = max_pair_index // len(self._potential_centers)
        max_row = max_pair_index % len(self._potential_centers)

        left_pole = self._potential_centers[max_col]
        right_pole = self._potential_centers[max_row]

        left_indexes, right_indexes = [], []

        if self.should_subsample:
            for i in range(0, len(self.points), self.batch_size):
                batch = self._get_batch(i)
                left_distances = numpy_calculate_distance(self.data[left_pole], batch, self.df)
                right_distances = numpy_calculate_distance(self.data[right_pole], batch, self.df)
                [(left_indexes if l < r else right_indexes).append(self.points[i + j])
                 for j, l, r in zip(range(len(batch)), left_distances, right_distances)]

        else:
            points = [self.data[p] for p in self.points]
            left_distances = numpy_calculate_distance(self.data[left_pole], points, self.df)
            right_distances = numpy_calculate_distance(self.data[right_pole], points, self.df)

            [(left_indexes if l < r else right_indexes).append(p)
             for l, r, p in zip(left_distances, right_distances, self.points)]

        self.left = Cluster(
            data=self.data,
            points=left_indexes,
            distance_function=self.df,
            name=f'{self.name}1',
            parent_lfd=self.lfd,
        )

        self.right = Cluster(
            data=self.data,
            points=right_indexes,
            distance_function=self.df,
            name=f'{self.name}2',
            parent_lfd=self.lfd,
        )

        return

    def make_tree(self) -> None:
        """ Build cluster tree starting at the current cluster. """
        self.pop()
        if self.left or self.right:
            self.left.make_tree()
            self.right.make_tree()
        return

    def search(self, query: np.ndarray, radius: float, search_depth: int) -> List[str]:
        """ Perform clustered search starting at the current cluster.

        :param query: point to search around.
        :param radius: search radius to consider.
        :param search_depth: maximum depth to which to search.

        :return: list of indexes of hits.
        """
        results = []
        if (self.depth < search_depth) and (self.left or self.right):
            if self.left.can_include(query, radius):
                results.extend(self.left.search(query, radius, search_depth))
            if self.right.can_include(query, radius):
                results.extend(self.right.search(query, radius, search_depth))
        else:
            results.append(self.name)
        return results

    def compress(self, filename):
        if self.left or self.right:
            return

        step = 10 ** (config.H_MAGNITUDE / (-2.5))
        center = self.data[self.center]
        points = [np.array(np.ceil((self.data[p] - center) / step), dtype=np.int64) for p in self.points]

        filename = f'{filename}/{self.name}.pickle'
        with open(filename, 'wb') as outfile:
            pickle.dump(points, outfile)
        return
