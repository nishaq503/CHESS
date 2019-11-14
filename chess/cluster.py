import pickle
from typing import List, Set

import numpy as np

from chess import globals
from chess.distance import calculate_distances


class Cluster:
    """ Defines the cluster class.
    Adds methods relevant to building and searching through a cluster-tree.
    Adds a compress method that should only be used for appropriate datasets.
    """

    def __init__(
            self,
            data: np.memmap,
            metric: str,
            points: Set[int] = None,
            name: str = '',
            center: int = None,
            radius: globals.RADII_DTYPE = None,
    ):
        """
        Initializes cluster object.

        :param data: numpy.memmap of points to cluster.
        :param points: list of indexes in data of the points in this cluster.
        :param metric: distance metric this cluster uses.
        :param name: name of cluster to track ancestry.
        :param center: index in data of center of cluster.
        :param radius: radius of cluster i.e. the maximum distance from any point in the cluster to the cluster center.
        """
        self.data: np.memmap = data
        self.name: str = name
        self.metric: str = metric
        self._center: int = center
        self._radius: globals.RADII_DTYPE = radius
        self.left = self.right = None
        self._samples: List[int] = None

        self.depth: int = len(name)

        self.points: List[int] = points or list(range(self.data.shape[0]))
        assert len(self.points) > 0, f"Empty point indices in {self.name}"
        assert len(self.points) == len(set(self.points)), f"Duplicate point indices in {self.name}"

        self._subsample: bool = len(self.points) > globals.SUBSAMPLING_LIMIT

    @property
    def center(self):
        if not self._samples:
            raise NotImplementedError("Gotta build first.")
        return self._samples[int(np.argmin(self._pairwise_distances().sum(axis=1)))]

    def _sample_unique(self):
        """ Returns indices of unique potential centers.

        # TODO: Examine cost of unique vs a loop on shuffled indices.
        """
        n = int(np.sqrt(len(self.points))) if self._subsample else len(self.points)
        unique = np.unique(self.data[self.points], return_index=True, axis=0)[1]
        return np.random.choice(unique, n, replace=False) if len(unique) > n else unique

    def _sample(self):
        """ Returns indices of potential centers. """
        n = int(np.sqrt(len(self.points))) if self._subsample else len(self.points)
        return np.random.choice(list(self.points), n, replace=False)

    def _pairwise_distances(self) -> np.ndarray:
        """ Calculates the pairwise distances between potential centers.

        :return: pairwise distance matrix of points that are potential centers.
        """
        points = self._sample()
        distances = calculate_distances(self.data[points], self.data[points], self.metric)

        if np.max(distances) == globals.RADII_DTYPE(0):
            # All sampled points were duplicates.
            points = self._sample_unique()
            distances = calculate_distances(self.data[points], self.data[points], self.metric)

        return distances

    def _iter_batch(self, batch_size: int = globals.BATCH_SIZE) -> np.ndarray:
        for i in range(0, len(self.points), batch_size):
            yield self.data[i:i + batch_size]

    def _batch(self, points: List[int], start_index: int, batch_size: int = globals.BATCH_SIZE) -> np.ndarray:
        """ Gets a batch of points from the given points list.

        Batch starts at index start_index in points list.

        :param points: list of indexes in self.data from which to draw the batch.
        :param start_index: index in points from where to start drawing the batch.
        :param batch_size: size of each batch.

        :return numpy array of points in the batch:
        """
        num_points = min(start_index + batch_size, len(points)) - start_index
        return np.asarray([self.data[p] for p in points[start_index: start_index + num_points]])

    def radius(self) -> globals.RADII_DTYPE:
        """ Calculates the radius of the cluster.

        This is the maximum of the distances of any point in the cluster to the cluster center.

        :return: radius of cluster.
        """
        if not self._radius:
            # TODO: Should radius be callable if center is None?
            center = self.data[self.center or 0]
            center = np.expand_dims(center, 0)
            radii = [np.max(calculate_distances(center, b, self.metric)) for b in self._iter_batch()]
            self._radius = np.max(radii)
            self._radius = self._radius if self._radius != globals.RADII_DTYPE(0.0) else globals.RADII_DTYPE(0.0)
            #
            # center = np.expand_dims(center, 0)
            #
            # if self._contains_only_duplicates:
            #     return globals.RADII_DTYPE(0.0)
            # potential_radii = [np.max(calculate_distances(center, self._batch(self.points, i), self.metric)[0, :])
            #                    for i in range(0, len(self.points), globals.BATCH_SIZE)]
            # radius = np.max(np.asarray(potential_radii, dtype=globals.RADII_DTYPE))
            # if not isinstance(radius, globals.RADII_DTYPE):
            #     raise ValueError(f'Got problem with calculating radius in cluster {self.name}.\n'
            #                      f'Radius was {radius}.')
        return self._radius

    def _calculate_local_fractal_dimension(self) -> globals.RADII_DTYPE:
        """
        Calculates the local fractal dimension of the cluster.
        This is the log2 ratio of the number of points in the cluster to the number of points within half the radius.

        :return: local fractal dimension of the cluster.
        """
        center = self.data[self.center]
        center = np.expand_dims(center, 0)

        if self._contains_only_duplicates:
            return globals.RADII_DTYPE(0)
        count = [1 if d <= (self._radius / 2) else 0
                 for i in range(0, len(self.points), globals.BATCH_SIZE)
                 for d in calculate_distances(center, self._batch(self.points, i), self.metric)[0, :]]
        count = np.sum(count, dtype=globals.RADII_DTYPE)
        return 0 if count == 0 else np.log2(globals.RADII_DTYPE(len(self.points)) / count)

    def can_include(
            self,
            query: np.ndarray,
            radius: globals.RADII_DTYPE = 0,
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

        return distance <= (self._radius + radius)

    def can_be_popped(self) -> bool:
        """
        Checks weather this cluster can be popped.

        :return: Weather or not the cluster can be popped.
        """
        return all((
            not self._contains_only_duplicates,
            globals.MIN_POINTS < len(self.points),
            globals.MIN_RADIUS < self._radius,
            self.depth < globals.MAX_DEPTH,
            not self.left,
            not self.right
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
        if update:
            self.update()

        max_pair_index = np.argmax(np.triu(self._pairwise_distances, k=1))
        max_col = max_pair_index // len(self._pairwise_distances)
        max_row = max_pair_index % len(self._pairwise_distances)

        left_pole_index = self._potential_centers[max_col]
        right_pole_index = self._potential_centers[max_row]

        if left_pole_index == right_pole_index:
            raise ValueError(f'Got the same point {right_pole_index} as both poles. Cluster name is {self.name}.\n')

        left_pole, right_pole = self.data[left_pole_index], self.data[right_pole_index]
        left_pole, right_pole = np.expand_dims(left_pole, 0), np.expand_dims(right_pole, 0)

        left_indexes, right_indexes = [], []

        def partition(points: np.ndarray, i):
            left_distances = calculate_distances(left_pole, points, self.metric)[0, :]
            right_distances = calculate_distances(right_pole, points, self.metric)[0, :]
            [(left_indexes if l < r else right_indexes).append(self.points[i + j])
             for j, l, r in zip(range(points.shape[0]), left_distances, right_distances)]
            return

        [partition(self._batch(self.points, i), i) for i in range(0, len(self.points), globals.BATCH_SIZE)]

        if left_pole_index in right_indexes:
            right_indexes.remove(left_pole_index)
            left_indexes.append(left_pole_index)
        if right_pole_index in left_indexes:
            left_indexes.remove(right_pole_index)
            right_indexes.append(right_pole_index)

        if len(left_indexes) == 0:
            raise ValueError(f'Got empty left_indexes after popping cluster {self.name}.\n')
        if len(right_indexes) == 0:
            raise ValueError(f'Got empty right_indexes after popping cluster {self.name}.\n')

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
            radius: globals.RADII_DTYPE,
            search_depth: int = globals.MAX_DEPTH,
    ) -> List[str]:
        """
        Perform clustered search from this cluster.

        :param query: point to search around.
        :param radius: search radius to consider.
        :param search_depth: maximum depth to which to search.

        :return: List of names of clusters that may contain hits.
        """
        results = []
        if self._radius <= radius:
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
