""" CHESS Cluster.

This is the underlying structure for CHESS.
"""
from typing import List, Tuple, Dict, Union, Callable

import numpy as np

from chess import globals
from chess.distance import calculate_distances
from chess.query import Query


class Cluster:
    """ Defines the cluster class.

    Adds methods relevant to building and searching through a cluster-tree.
    Adds a compress method that should only be used for appropriate datasets.
    """

    def __init__(
            self,
            data: Union[np.memmap, np.ndarray],
            metric: str,
            points: List[int] = None,
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
        # Required from constructor, either from user or as defaults.
        self.data: np.memmap = data
        self.metric: str = metric
        self.name: str = name
        self._radius: globals.RADII_DTYPE = radius

        # Provided or computed values. (cached)
        self.points: List[int] = points or list(range(self.data.shape[0]))
        self.subsample: bool = len(self.points) > globals.SUBSAMPLING_LIMIT
        self.samples, self.distances = self._samples()
        self.center = center or self._center()
        self.depth: int = len(name)
        self._all_same: bool = np.max(self.distances) == globals.RADII_DTYPE(0.0)

        # Children.
        self.left = self.right = None

        # Invariants after construction.
        assert len(self.points) > 0, f"Empty point indices in {self.name}"
        assert len(self.points) == len(set(self.points)), f"Duplicate point indices in cluster {self.name}:\n{self.points}"
        assert self.center is not None

    def __iter__(self):
        """ Iterates over points within the cluster. """
        for i in range(0, len(self.points), globals.BATCH_SIZE):
            yield self.data[self.points[i:i + globals.BATCH_SIZE]]

    def __len__(self) -> int:
        """ Returns the number of points within the cluster. """
        return len(self.points)

    def __getitem__(self, item):
        """ Gets the point at the index. """
        return self.data[self.points[item]]

    def __contains__(self, query: Query):
        """ Determines whether or not a query falls into the cluster. """
        center = np.expand_dims(self.data[self.center], 0)
        distance = calculate_distances(center, [query.point], self.metric)[0, 0]
        return distance <= (self.radius() + query.radius)

    def __str__(self):
        return f'{self.name}, [{", ".join([str(p) for p in self.points])}]'

    def __repr__(self):
        return ','.join(map(str, [
            self.name,
            len(self.points),
            self.center,
            self.radius(),
            self.local_fractal_dimension(),
            self.partitionable(),
        ]))

    def __eq__(self, other):
        return self.metric == other.metric \
               and self.points == other.points \
               and np.all(self.data == other.data)

    def dict(self):
        d: Dict[str: Cluster]
        d = {c.name: c for c in self.inorder()}
        return d

    def _samples(self) -> Tuple[List[int], np.ndarray]:
        """ Returns the possible centroids and their pairwise distances.

        It ensures that the possible centroids selected cannot all be the same point.

        # TODO: Compare the overhead of calling unique, should we just do that every time?
        """
        n = int(np.sqrt(len(self.points))) if self.subsample else len(self.points)
        # First, just try to grab n points.
        points = np.random.choice(list(self.points), n, replace=False)
        distances = calculate_distances(self.data[points], self.data[points], self.metric)

        if np.max(distances) == globals.RADII_DTYPE(0.0):
            # If all sampled points were duplicates, we grab non-duplicate centroids.
            unique = np.unique(self.data[self.points], return_index=True, axis=0)[1]
            points = np.random.choice(unique, n, replace=False) if len(unique) > n else unique
            distances = calculate_distances(self.data[points], self.data[points], self.metric)
            # TODO: Should there be any checks here, like max(distance)?

        return points, distances

    def _center(self):
        """ Returns the index of the centroid of the cluster."""
        return self.samples[int(np.argmin(self.distances.sum(axis=1)))]

    def radius(self) -> globals.RADII_DTYPE:
        """ Calculates the radius of the cluster.

        This is the maximum of the distances of any point in the cluster to the cluster center.

        :return: radius of cluster.
        """
        if not self._radius and not self._all_same:
            assert self.center is not None
            center = self.data[self.center]
            center = np.expand_dims(center, 0)
            radii = [np.max(calculate_distances(center, b, self.metric)) for b in self]
            self._radius = np.max(radii)
            self._radius = globals.RADII_DTYPE(self._radius)

        return self._radius or globals.RADII_DTYPE(0.0)

    def local_fractal_dimension(self) -> globals.RADII_DTYPE:
        """ Calculates the local fractal dimension of the cluster.
        This is the log2 ratio of the number of points in the cluster to the number of points within half the radius.

        :return: local fractal dimension of the cluster.
        """
        center = self.data[self.center]
        center = np.expand_dims(center, 0)

        if self._all_same:
            return globals.RADII_DTYPE(0.0)
        count = [d <= (self.radius() / 2)
                 for batch in self
                 for d in calculate_distances(center, batch, self.metric)[0:]]
        count = np.sum(count, dtype=globals.RADII_DTYPE)
        return count or np.log2(globals.RADII_DTYPE(len(self.points)) / count)

    def partitionable(self) -> bool:
        """ Returns weather or not this cluster can be partitioned.
        """
        return all((
            not self._all_same,
            not (self.left or self.right),
            self.depth < globals.MAX_DEPTH,
            len(self.points) > globals.MIN_POINTS,
            self.radius() > globals.MIN_RADIUS,
        ))

    def partition(self):
        """ Partition this cluster into left and right children.

        Steps:
            * Check if the cluster has already been popped or if it can be popped.
            * Find the two potential centers that are the farthest apart.
            * Treat those two as the left and right poles.
            * Partition the points in this cluster by the pole that the points are closer to.
            * Assign the partitioned points to the left and right child clusters appropriately.
        """
        max_pair_index = np.argmax(np.triu(self.distances, k=1))
        max_col = max_pair_index // len(self.distances)
        max_row = max_pair_index % len(self.distances)

        left_pole_index = self.samples[max_col]
        right_pole_index = self.samples[max_row]

        assert left_pole_index != right_pole_index, f'Left pole and right pole are equal in cluster {self.name}'

        left_pole, right_pole = self.data[left_pole_index], self.data[right_pole_index]
        left_pole, right_pole = np.expand_dims(left_pole, 0), np.expand_dims(right_pole, 0)

        left_indices, right_indices = [], []
        for i, batch in enumerate(self):
            # Loop invariant: lists of indices will hold no duplicates.
            assert len(left_indices) == len(set(left_indices)), f'Left indices of cluster {self.name} contains duplicates.'
            assert len(right_indices) == len(set(right_indices)), f'Right indices of cluster {self.name} contains duplicates.'

            left_dist = calculate_distances(left_pole, batch, self.metric)[0, :]
            right_dist = calculate_distances(right_pole, batch, self.metric)[0, :]
            for j, l, r in zip(range(len(batch)), left_dist, right_dist):
                (left_indices if l < r else right_indices).append(self.points[i * globals.BATCH_SIZE + j])

        # Loop termination invariant: there are points in each half.
        assert len(left_indices) > 0, f'Empty left cluster after partitioning {self.name}'
        assert len(right_indices) > 0, f'Empty right cluster after partitioning {self.name}'

        self.left = Cluster(
            data=self.data,
            metric=self.metric,
            points=left_indices,
            name=f'{self.name}0',
        )

        self.right = Cluster(
            data=self.data,
            metric=self.metric,
            points=right_indices,
            name=f'{self.name}1',
        )
        return

    def make_tree(self, stopping_criteria: Callable[[any], bool] = None):
        """
        Build cluster sub-tree starting at this cluster.
        """
        if stopping_criteria:
            if stopping_criteria(self):
                self.partition()

        elif self.partitionable():
            self.partition()

        if self.left:
            self.left.make_tree()
        if self.right:
            self.right.make_tree()
        return

    def compress(self):
        """ Compresses a leaf cluster.
        # TODO: Migrate away from pickle. Perhaps we can build a new memmap?
        """
        assert not (self.left or self.right), f'Can only compress leaves! Tried to compress {self.name}.'

        step_size = 10 ** (globals.H_MAGNITUDE / (-2.5))
        center = self.data[self.center]
        points = [np.asarray(np.ceil((self.data[p] - center) // step_size), dtype=np.int64)
                  for p in self.points]
        return points

    ###################################
    # Traversals
    ###################################
    def inorder(self):
        return self._inorder(self)

    def _inorder(self, node):
        if node.left:
            for n in self._inorder(node.left):
                yield n
        yield node
        if node.right:
            for n in self._inorder(node.right):
                yield n

    def postorder(self):
        return self._postorder(self)

    def _postorder(self, node):
        if node.left:
            for n in self._postorder(node.left):
                yield n
        if node.right:
            for n in self._postorder(node.right):
                yield n
        yield node

    def preorder(self):
        return self._preorder(self)

    def _preorder(self, node):
        yield node
        if node.left:
            for n in self._preorder(node.left):
                yield n
        if node.right:
            for n in self._preorder(node.right):
                yield n

    def leaves(self):
        return self._leaves(self)

    def _leaves(self, node):
        if not (node.left or node.right):
            yield node

        if node.left:
            for n in self._leaves(node.left):
                yield n

        if node.right:
            for n in self._leaves(node.right):
                yield n
