import pickle
from typing import List, Tuple, Dict, Union

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

    def _center(self):
        """ Returns the index of the centroid of the cluster."""
        return self.samples[int(np.argmin(self.distances.sum(axis=1)))]

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
        distance = calculate_distances(center, query.point, self.metric)[0, 0]
        return distance <= (self.radius() + query.radius)

    def __str__(self):
        return ', '.join([self.name, *[str(p) for p in self.points]])

    def __repr__(self):
        return ','.join([
            self.name,
            len(self.points),
            self.center,
            self.radius(),
            self.local_fractal_dimension(),
            self.partitionable,
        ])

    def dict(self):
        d = Dict[str: Cluster] = {}

        def inorder(node: Cluster):
            if node.left:
                inorder(node.left)
            d[node.name] = node
            if node.right:
                inorder(node.right)

        inorder(self)
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

    def _iter_batch(self, start_index: int = 0, batch_size: int = globals.BATCH_SIZE) -> np.ndarray:
        """ Iterates over batches of points from the given points list.

        Batch starts at index start_index in points list.

        :param start_index: index in points from where to start drawing the batch.
        :param batch_size: size of each batch.

        :return numpy array of points in the batch:
        """
        for i in range(start_index, len(self), batch_size):
            yield self[i:i + batch_size]

    def radius(self) -> globals.RADII_DTYPE:
        """ Calculates the radius of the cluster.

        This is the maximum of the distances of any point in the cluster to the cluster center.

        :return: radius of cluster.
        """
        if not self._radius and not self._all_same:
            assert self.center
            center = self.data[self.center]
            center = np.expand_dims(center, 0)
            radii = [np.max(calculate_distances(center, b, self.metric)) for b in self]
            self._radius = np.max(radii)
            # TODO: What does this line do?
            self._radius = self._radius if self._radius != globals.RADII_DTYPE(0.0) else globals.RADII_DTYPE(0.0)

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
        assert self.partitionable()

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
            left_dist = calculate_distances(left_pole, batch, self.metric)[0, :]
            right_dist = calculate_distances(right_pole, batch, self.metric)[0, :]
            for j, l, r in zip(range(len(batch)), left_dist, right_dist):
                (left_indices if l < r else right_indices).append(self.points[i * globals.BATCH_SIZE + j])

        assert len(left_indices) == len(set(left_indices))
        assert len(right_indices) == len(set(right_indices))

        def partition(points: np.ndarray):
            left_distances = calculate_distances(left_pole, points, self.metric)[0, :]
            right_distances = calculate_distances(right_pole, points, self.metric)[0, :]
            [(left_indices if l < r else right_indices).append(self.points[j])
             for j, l, r in zip(range(points.shape[0]), left_distances, right_distances)]
            return

        # [partition(batch) for batch in self]

        if left_pole_index in right_indices:
            right_indices.remove(left_pole_index)
            left_indices.append(left_pole_index)
        if right_pole_index in left_indices:
            left_indices.remove(right_pole_index)
            right_indices.append(right_pole_index)

        if len(left_indices) == 0:
            raise ValueError(f'Got empty left_indexes after partitioning cluster {self.name}.\n')
        if len(right_indices) == 0:
            raise ValueError(f'Got empty right_indexes after partitioning cluster {self.name}.\n')

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

    def make_tree(self):
        """
        Build cluster sub-tree starting at this cluster.
        """
        self.partition()
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

    def print_names(
            self,
            filename: str = None
    ):
        """
        Write a .csv with columns {cluster_name, point_index}.
        :param filename: Optional file to write to.
        """
        filename = self.names_file if filename is None else filename
        with open(filename, 'w') as outfile:
            outfile.write('cluster_name,point\n')

            def _helper(node: Cluster):
                if node.left:
                    _helper(node.left)
                if node.right:
                    _helper(node.right)
                else:
                    [outfile.write(f'{node.name},{str(p)}\n') for p in node.points]
                    outfile.flush()

            _helper(self.root)
        return

    def print_info(
            self,
            filename: str = None
    ):
        """
        Write a .csv with columns {cluster_name, number_of_points, center, radius, lfd, is_leaf}.
        :param filename: Optional file to write to.
        """
        filename = self.info_file if filename is None else filename
        with open(filename, 'w') as outfile:
            outfile.write('cluster_name,number_of_points,center,radius,lfd,is_leaf\n')

            def _helper(node: Cluster):
                outfile.write(f'{node.name},{len(node.points):d},{node.center:d},{node._radius:.8f},'
                              f'{node.local_fractal_dimension:.8f},{not node.partitionable()}\n')
                if node.left:
                    _helper(node.left)
                if node.right:
                    _helper(node.right)

            _helper(self.root)
        return

    def _build_dict_tree(name: str) -> List[int]:
        if name in name_to_points.keys():
            return name_to_points[name]
        else:
            left = _build_dict_tree(f'{name}0')
            right = _build_dict_tree(f'{name}1')
            name_to_points[name] = left.copy() + right.copy()
            return name_to_points[name]

        _build_dict_tree('')

        self.info_file = info_file if info_file is not None else self.info_file
        name_to_info: Dict[str, List] = {}
        with open(self.info_file, 'r') as infile:
            infile.readline()
            while True:
                line = infile.readline()
                if not line:
                    break

                [cluster_name, num_points, center, radius, lfd, is_leaf] = line.split(',')
                if cluster_name not in name_to_points.keys():
                    raise ValueError(f'{cluster_name} not found in name_to_points dictionary.')

                name_to_info[cluster_name] = [int(center), globals.RADII_DTYPE(radius),
                                              globals.RADII_DTYPE(lfd), bool(is_leaf)]
                if len(name_to_points[cluster_name]) != int(num_points):
                    raise ValueError(f'Mismatch in number of points in cluster {cluster_name}.\n'
                                     f'Got {num_points} from {self.info_file}.\n'
                                     f'Got {len(name_to_points[cluster_name])} from {self.names_file}.')

        def _build_tree(name: str) -> Cluster:
            return Cluster(
                data=self.data,
                points=name_to_points[name].copy(),
                metric=self.metric,
                name=name,
                center=name_to_info[name][0],
                radius=name_to_info[name][1],
                local_fractal_dimension=name_to_info[name][2],
                left=_build_tree(f'{name}0'),
                right=_build_tree(f'{name}1'),
                reading=True,
            ) if name in name_to_points else None

        return _build_tree('')

    def cluster_deeper(
            self,
            new_depth: int,
    ):
        old_depth = max(list(map(len, self.cluster_dict.keys())))

        [cluster.partition(update=True)
         for i in range(old_depth, new_depth)
         for name, cluster in self.cluster_dict.items()
         if len(name) == i]

        self.cluster_dict = self._get_cluster_dict()
        return

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
