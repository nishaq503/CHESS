from typing import List

import numpy as np

import config
from src.utils import calculate_pairwise_distances, simple_distance, calculate_distance


class Cluster:
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
        self.name: str = name
        self.depth: int = len(self.name)

        self.data: np.memmap = data
        self.points: List[int] = points
        self.distance_function: str = distance_function
        self.parent_lfd: float = parent_lfd
        self.left: Cluster = left
        self.right: Cluster = right

        self.batch_size = config.BATCH_SIZE
        self.max_depth = config.MAX_DEPTH
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

    def _get_potential_centers(self):
        if self.should_subsample:
            sample_size = int(np.sqrt(len(self.points)))
            points = self.points.copy()
            np.random.shuffle(points)
            return points[: sample_size + 1]
        else:
            return self.points.copy()

    def _calculate_pairwise_distances(self):
        if self.should_subsample:
            points = np.asarray([self.data[p] for p in self._potential_centers])
            return calculate_pairwise_distances(a=points, distance_function=self.distance_function)
        else:
            points = np.asarray([self.data[p] for p in self.points])
            return simple_distance(a=points, b=points, distance_function=self.distance_function)

    def _calculate_center(self):
        sum_distances = np.sum(a=self._pairwise_distances, axis=1)
        return self._potential_centers[int(np.argmin(sum_distances))]

    def _get_batch(self, start):
        num_points = min(start + self.batch_size, len(self.points)) - start
        return np.asarray([self.data[p] for p in self.points[start: start + num_points]])

    def _calculate_radius(self):
        if self.should_subsample:
            return max(max([max(calculate_distance(self.data[self.center], self._get_batch(i), self.distance_function))
                            for i in range(0, len(self.points), self.batch_size)]), 0.0)
        else:
            points = np.asarray([self.data[p] for p in self.points])
            distances = simple_distance(a=self.data[self.center], b=points, distance_function=self.distance_function)
            return max(max(distances), 0.0)

    def _calculate_lfd(self):
        if self.should_subsample:
            count = sum([sum([1 if d < self.radius / 2 else 0
                              for d in calculate_distance(self.data[self.center],
                                                          self._get_batch(i),
                                                          self.distance_function)])
                         for i in range(0, len(self.points), self.batch_size)])
        else:
            points = np.asarray([self.data[p] for p in self.points])
            distances = simple_distance(self.data[self.center], points, self.distance_function)
            count = sum([1 if d < self.radius / 2 else 0 for d in distances])

        return 0 if count == 0 else np.log2(len(self.points) / count)

    def update(self):
        self._potential_centers: List[int] = self._get_potential_centers()
        self._pairwise_distances: np.ndarray = self._calculate_pairwise_distances()

        self.center: int = self._calculate_center()
        self.radius: float = self._calculate_radius()
        self.lfd: float = self._calculate_lfd()
        return

    def can_include(self, query: np.ndarray, radius: float = 0):
        distance = simple_distance(query, self.data[self.center], self.distance_function)
        return distance < (self.radius + radius)

    def can_be_popped(self):
        return all((
            len(self.points) > self.min_points,
            self.radius > self.min_radius,
            self.depth < self.max_depth,
            self.lfd < config.LFD_LIMIT or self.lfd > self.parent_lfd,
        ))

    def pop(self):
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
                left_distances = simple_distance(self.data[left_pole], batch, self.distance_function)
                right_distances = simple_distance(self.data[right_pole], batch, self.distance_function)
                [(left_indexes if l < r else right_indexes).append(self.points[i + j])
                 for j, l, r in zip(range(len(batch)), left_distances, right_distances)]

        else:
            points = [self.data[p] for p in self.points]
            left_distances = simple_distance(self.data[left_pole], points, self.distance_function)
            right_distances = simple_distance(self.data[right_pole], points, self.distance_function)

            [(left_indexes if l < r else right_indexes).append(p)
             for l, r, p in zip(left_distances, right_distances, self.points)]

        self.left = Cluster(
            data=self.data,
            points=left_indexes,
            distance_function=self.distance_function,
            name=f'{self.name}1',
            parent_lfd=self.lfd,
        )

        self.right = Cluster(
            data=self.data,
            points=right_indexes,
            distance_function=self.distance_function,
            name=f'{self.name}2',
            parent_lfd=self.lfd,
        )

        return

    def make_tree(self):
        self.pop()
        if self.left or self.right:
            self.left.make_tree()
            self.right.make_tree()
        return

    def search(self, query: np.ndarray, radius: float, max_depth: int) -> List[int]:
        hits = []
        if (self.depth < max_depth) and (self.left or self.right):
            if self.left.can_include(query, radius):
                hits.extend(self.left.search(query, radius, max_depth))
            if self.right.can_include(query, radius):
                hits.extend(self.right.search(query, radius, max_depth))
        else:
            if self.should_subsample:
                for i in range(0, len(self.points), self.batch_size):
                    batch = self._get_batch(i)
                    distances = calculate_distance(query, batch, self.distance_function)
                    hits.extend([self.points[i + j] for j, d in enumerate(distances) if d <= radius])
            else:
                points = [self.data[p] for p in self.points]
                distances = simple_distance(query, points, self.distance_function)
                hits.extend([self.points[j] for j, d in enumerate(distances) if d <= radius])
        return hits
