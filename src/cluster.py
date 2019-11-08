import pickle
from collections import Counter
from typing import List

import numpy as np

from src import globals
from src.distance_functions import calculate_distances


class Cluster:
    """
    Defines the cluster class.
    Adds methods relevant to building and searching through a cluster-tree.
    Adds a compress method that should only be used for apogee data with euclidean distance.
    """

    def __init__(
            self,
            data: np.memmap,
            points: List[int],
            metric: str,
            name: str,
            center: int = None,
            radius: globals.FLOAT_DTYPE = None,
            local_fractal_dimension: globals.FLOAT_DTYPE = None,
            left=None,
            right=None,
            reading: bool = False,
    ):
        """
        Initializes cluster object.

        :param data: numpy.memmap of points to cluster.
        :param points: list of indexes in data of the points in this cluster.
        :param metric: distance metric this cluster uses.
        :param name: name of cluster to track ancestry.
        :param center: index in data of center of cluster.
        :param radius: radius of cluster i.e. the maximum distance from any point in the cluster to the cluster center.
        :param local_fractal_dimension: log2 ratio of number of points in cluster to number points within half radius.
        :param left: left-child cluster.
        :param right: right-child cluster.
        :param reading: weather or not a cluster is being read from file.
        """

        self.data: np.memmap = data
        self.name: str = name
        self.depth: int = len(name)

        error_suffix = f'\nCluster name is {self.name}.'
        if len(points) == 0:
            raise ValueError(f'points list must have at least one point.'
                             f'Cluster name is {self.name}.' + error_suffix)
        if len(points) != len(set(points)):
            duplicates = [(p, count) for p, count in Counter(points).items() if count > 1]
            raise ValueError(f'points list must not have duplicate elements. '
                             f'These points were duplicated:\n{duplicates}.' + error_suffix)
        self.points: List[int] = points

        if metric not in globals.DISTANCE_FUNCTIONS:
            raise NotImplementedError(f'Got metric {metric}. It must be one of {globals.DISTANCE_FUNCTIONS}.')
        self.metric: str = metric

        self.left: Cluster = left
        self.right: Cluster = right

        self._should_subsample_centers: bool = len(points) > globals.SUBSAMPLING_LIMIT
        self._potential_centers: List[int] = []
        self._pairwise_distances: np.ndarray = np.asarray([[]])

        self.center: int
        self.radius: globals.FLOAT_DTYPE
        self.local_fractal_dimension: globals.FLOAT_DTYPE

        if len(self.points) == 1:
            self.center = self.points[0]
            self.radius = 0
            self.local_fractal_dimension = 0
        elif reading:
            self.center = center
            self.radius = radius
            self.local_fractal_dimension = local_fractal_dimension
        else:
            self.update()

    def _get_potential_centers(self) -> List[int]:
        """
        If the cluster contains too many points, subsample square_root as many points as potential centers.
        Otherwise any point in the points list is a potential center.
        :return: list of indexes in self.data of points that could be the center of the cluster.
        """
        points: List[int] = self.points.copy()

        if self._should_subsample_centers:
            num_samples = int(np.sqrt(len(self.points)))
            np.random.shuffle(points)
            points = points[: num_samples + 1]

        return points

    def _calculate_pairwise_distances(self) -> np.ndarray:
        """
        Calculates the pairwise distances between potential centers.

        :return: pairwise distance matrix of points that are potential centers.
        """
        points = np.asarray([self.data[p] for p in self._potential_centers])
        return calculate_distances(points, points, self.metric)

    def _calculate_center(self) -> int:
        """
        Calculates the geometric median of the potential centers.
        This point is the center of the cluster.

        :return: index in self.data of the center of the cluster.
        """
        return self._potential_centers[int(np.argmin(self._pairwise_distances.sum(axis=1)))]

    def _get_batch(
            self,
            points: List[int],
            start_index: int,
    ) -> np.ndarray:
        """
        Gets a batch of points from the given points list.
        Batch size is globals.BATCH_SIZE.
        Batch starts at index start_index in points list.

        :param points: list of indexes in self.data from which to draw the batch.
        :param start_index: index in points from where to start drawing the batch.

        :return numpy array of points in the batch:
        """
        num_points = min(start_index + globals.BATCH_SIZE, len(points)) - start_index
        return np.asarray([self.data[p]
                           for p in points[start_index: start_index + num_points]])

    # noinspection PyTypeChecker
    def _calculate_radius(self) -> globals.FLOAT_DTYPE:
        """
        Calculates the radius of the cluster.
        This is the maximum of the distances of any point in the cluster to the cluster center.

        :return: radius of cluster.
        """
        center = self.data[self.center]
        center = np.expand_dims(center, 0)

        if len(self.points) > globals.BATCH_SIZE:
            return max(
                max([max(calculate_distances(center,
                                             self._get_batch(self.points, i),
                                             self.metric)[0, :])
                     for i in range(0, len(self.points), globals.BATCH_SIZE)]
                    ),
                0.0
            )
        else:
            return max(
                max(calculate_distances(
                        center,
                        self._get_batch(self.points, 0),
                        self.metric)[0, :]),
                0.0
            )

    def _calculate_local_fractal_dimension(self) -> globals.FLOAT_DTYPE:
        """
        Calculates the local fractal dimension of the cluster.
        This is the log2 ratio of the number of points in the cluster to the number of points within half the radius.

        :return: local fractal dimension of the cluster.
        """

        count: int
        center = self.data[self.center]
        center = np.expand_dims(center, 0)

        if len(self.points) > globals.BATCH_SIZE:
            count = sum(
                [sum(
                    [1 if d <= self.radius / 2 else 0
                     for d in calculate_distances(center,
                                                  self._get_batch(self.points, i),
                                                  self.metric)[0, :]]
                ) for i in range(0, len(self.points), globals.BATCH_SIZE)]
            )
        else:
            count = sum([1 if d <= self.radius / 2 else 0
                         for d in calculate_distances(center,
                                                      self._get_batch(self.points, 0),
                                                      self.metric)[0, :]])
        return 0 if count == 0 else np.log2(len(self.points) / count)

    def update(
            self,
            internals_only: bool = False,
    ):
        """
        Updates the relevant private variables of the cluster.

        :param internals_only: Weather or not to update the variables that are costly to update.
        """
        self._potential_centers = self._get_potential_centers()
        self._pairwise_distances = self._calculate_pairwise_distances()

        if not internals_only:
            self.center = self._calculate_center()
            self.radius = self._calculate_radius()
            self.local_fractal_dimension = self._calculate_local_fractal_dimension()

        return

    def can_include(
            self,
            query: np.ndarray,
            radius: globals.FLOAT_DTYPE = 0,
    ) -> bool:
        """
        Checks weather or not the given query can ne included in the cluster.

        :param query: point to check for possible inclusion.
        :param radius: if the point is a search query, add this radius to self.radius.

        :return: Weather or not the query falls within the required distance from the cluster center.
        """
        center = self.data[self.center]
        center = np.expand_dims(center, 0)
        distance = calculate_distances(center, query, self.metric)[0, 0]

        return distance <= (self.radius + radius)

    def can_be_popped(self) -> bool:
        """
        Checks weather this cluster can be popped.

        :return: Weather or not the cluster can be popped.
        """
        return all((
            len(self.points) > globals.MIN_POINTS,
            self.radius > globals.MIN_RADIUS,
            self.depth < globals.MAX_DEPTH,
        ))

    def pop(
            self,
            update: bool = False,
    ):
        """
        Pop this cluster into left and right children.

        Steps:
            * Check if the cluster has already been popped or if it can be popped.
            * Find the two potential centers that are the farthest apart.
            * Treat those two as the left and right poles.
            * Partition the points in this cluster by the pole that the points are closer to.
            * Assign the partitioned points to the left and right child clusters appropriately.

        :param update: weather or not up update internals before popping cluster.
        """

        if self.left or self.right or (not self.can_be_popped()):
            return

        if update:
            self.update(internals_only=True)

        max_pair_index = np.argmax(self._pairwise_distances)
        max_col = max_pair_index // len(self._potential_centers)
        max_row = max_pair_index % len(self._potential_centers)

        left_pole_index = self._potential_centers[max_col]
        right_pole_index = self._potential_centers[max_row]

        left_pole, right_pole = self.data[left_pole_index], self.data[right_pole_index]
        left_pole, right_pole = np.expand_dims(left_pole, 0), np.expand_dims(right_pole, 0)

        left_indexes, right_indexes = [], []

        def partition(points: np.ndarray, i):
            left_distances = calculate_distances(left_pole, points, self.metric)[0, :]
            right_distances = calculate_distances(right_pole, points, self.metric)[0, :]
            [(left_indexes if l < r else right_indexes).append(self.points[i + j])
             for j, l, r in zip(range(points.shape[0]), left_distances, right_distances)]
            return

        if len(self.points) > globals.BATCH_SIZE:
            [partition(self._get_batch(self.points, i), i)
             for i in range(0, len(self.points), globals.BATCH_SIZE)]
        else:
            partition(self._get_batch(self.points, 0), 0)

        self.left = Cluster(
            data=self.data,
            points=left_indexes,
            metric=self.metric,
            name=f'{self.name}0',
        )

        self.right = Cluster(
            data=self.data,
            points=right_indexes,
            metric=self.metric,
            name=f'{self.name}1',
        )
        return

    def make_tree(self):
        """
        Build cluster sub-tree starting at this cluster.
        """
        self.pop()
        if self.left:
            self.left.make_tree()
        if self.right:
            self.right.make_tree()
        return

    def search(
            self,
            query: np.ndarray,
            radius: globals.FLOAT_DTYPE,
            search_depth: int,
    ) -> List[str]:
        """
        Perform clustered search from this cluster.

        :param query: point to search around.
        :param radius: search radius to consider.
        :param search_depth: maximum depth to which to search.

        :return: List of names of clusters that may contain hits.
        """
        results = []
        if self.radius <= radius:
            results.append(self.name)
        elif (self.depth < search_depth) and (self.left or self.right):
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

        step_size = 10 ** (globals.H_MAGNITUDE / (-2.5))
        center = self.data[self.center]
        points = [np.asarray(np.ceil((self.data[p] - center) // step_size), dtype=np.int64)
                  for p in self.points]

        filepath = f'{filename}/{self.name}.pickle'
        with open(filepath, 'wb') as outfile:
            pickle.dump(points, outfile)
        return
