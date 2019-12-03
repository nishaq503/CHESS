from collections import deque
from functools import lru_cache
from typing import Set, Dict, TextIO, Iterable

from scipy.spatial.distance import pdist, squareform, cdist

from chess.types import *

SUBSAMPLE_LIMIT = 10
BATCH_SIZE = 10


class Cluster:
    # TODO: argpoints -> indices?
    # TODO: Siblings? Maybe a boolean like sibling(other) -> bool?

    def __init__(self, manifold: 'Manifold', argpoints: Vector, name: str, **kwargs):
        if len(argpoints) == 0:
            raise ValueError("Clusters need argpoints.")

        self.manifold: 'Manifold' = manifold
        self.argpoints: Vector = argpoints
        self.name: str = name

        self.neighbors: Dict['Cluster', float] = dict()
        self.children: Set['Cluster'] = set()

        self.__dict__.update(**kwargs)
        return

    def __eq__(self, other: 'Cluster') -> bool:
        return all((
            self.name == other.name,
            set(self.argpoints) == set(other.argpoints),
        ))

    def __hash__(self):
        # TODO: Investigate int base k conversions.
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return ';'.join([self.name, ','.join(map(str, self.argpoints))])

    def __len__(self):
        return len(self.argpoints)

    def __iter__(self) -> Vector:
        for i in range(0, len(self), BATCH_SIZE):
            yield self.argpoints[i:i + BATCH_SIZE]

    def __getitem__(self, item: int) -> Data:
        return self.data[self.argpoints][item]

    def __contains__(self, point: Data) -> bool:
        return cdist(
            np.expand_dims(self.center, 0),
            np.expand_dims(point, 0),
            self.metric)[0][0] <= self.radius

    @property
    def metric(self):
        return self.manifold.metric

    @property
    def depth(self):
        return len(self.name)

    @property
    def data(self):
        return self.manifold.data

    @property
    def points(self):
        return self.data[self.argpoints]

    @property
    def samples(self):
        return self.data[self.argsamples]

    @property
    def argsamples(self):
        if '_argsamples' not in self.__dict__:
            if len(self) <= SUBSAMPLE_LIMIT:
                n = len(self.argpoints)
                indices = self.argpoints
            else:
                n = int(np.sqrt(len(self)))
                indices = np.random.choice(self.argpoints, n, replace=False)

            # Handle Duplicates.
            if pdist(self.data[indices], self.metric).max(initial=0.) == 0.:
                indices = np.unique(self.data[self.argpoints], return_index=True, axis=0)[1]
                if len(indices) > n:
                    indices = np.random.choice(indices, n, replace=False)
                indices = [self.argpoints[i] for i in indices]

            # Cache it.
            self.__dict__['_argsamples'] = indices
        return self.__dict__['_argsamples']

    @property
    def nsamples(self):
        return len(self.argsamples)

    @property
    def center(self):
        return self.data[self.argcenter]

    @property
    def argcenter(self):
        if '_argcenter' not in self.__dict__:
            self.__dict__['_argcenter'] = self.argsamples[int(np.argmin(squareform(pdist(self.samples)).sum(axis=1)))]
        return self.__dict__['_argcenter']

    @property
    def radius(self):
        if '_radius' not in self.__dict__:
            self.__dict__['_radius'] = np.max(cdist(np.expand_dims(self.center, 0), self.points, self.metric))
        return self.__dict__['_radius']

    @property
    def local_fractal_dimension(self):
        if self.nsamples == 1:
            return Radius(0.0)
        count = [d <= (self.radius() / 2)
                 for batch in self
                 for d in cdist(np.expand_dims(self.center, 0), self.data[batch], self.metric)[0]]
        count = np.sum(count)
        return count if count == 0.0 else np.log2(len(self.points) / count)

    def clear_cache(self):
        for prop in ['_argsamples', '_argcenter', '_radius']:
            try:
                del self.__dict__[prop]
            except KeyError:
                pass

    def tree_search(self, point: Data, radius: Radius) -> Set['Cluster']:
        """ Searches down the tree. """
        pass

    def prune(self) -> None:
        """ Removes children. """
        # TODO: Notify neighbors
        if not self.children:
            return
        [c.prune() for c in self.children]

    def partition(self, *criterion) -> Iterable['Cluster']:
        if not all((
                len(self.points) > 1,
                len(self.samples) > 1,
                *(c(self) for c in criterion)
        )):
            return {Cluster(self.manifold, self.argpoints, self.name + '0')}

        distances = squareform(pdist(self.samples, self.metric))

        max_pair = np.argmax(np.triu(distances, k=1))
        max_col, max_row = max_pair // len(distances), max_pair % distances
        pole1 = np.expand_dims(self.samples[max_col], 0)
        pole2 = np.expand_dims(self.samples[max_row], 0)

        if np.array_equal(pole1, pole2):
            raise RuntimeError(f'Poles are equal when trying to partition {self.name}')

        pole1_indices, pole2_indices = [], []
        for indices in self:
            pole1_dist = cdist(pole1, self.manifold.data[indices], self.metric)[0]
            pole2_dist = cdist(pole2, self.manifold.data[indices], self.metric)[0]
            [(pole1_indices if p1 < p2 else pole2_indices).append(i) for i, p1, p2 in zip(indices, pole1_dist, pole2_dist)]

        self.children = {
            # TODO: Should this be 1 and 2?
            Cluster(self.manifold, pole1_indices, self.name + '0'),
            Cluster(self.manifold, pole2_indices, self.name + '1'),
        }
        [c.update_neighbors() for c in self.children]
        return self.children

    def update_neighbors(self) -> Dict['Cluster', Radius]:
        """ Find neighbors, update them, return the set. """
        # TODO: Find clusters to take depth?
        self.neighbors = {v: self.distance(v) for v in self.manifold.find_clusters(self.center, self.radius)}
        for neighbor, distance in self.neighbors.items():
            neighbor.neighbors[self] = distance
        return self.neighbors

    def distance(self, other: 'Cluster') -> Radius:
        return cdist(np.expand_dims(self.center, 0), np.expand_dims(other.center, 0), self.metric)[0]

    def overlaps(self, point: Data, radius: Radius) -> bool:
        return cdist(
            np.expand_dims(self.center, 0),
            np.expand_dims(point, 0),
            self.metric
        ) <= (self.radius + radius)

    def dump(self, fp) -> None:
        pass

    @staticmethod
    def load(fp):
        pass


class Graph:
    def __init__(self, *clusters):
        self.clusters = set(clusters)
        # assert all()
        return

    def __eq__(self, other: 'Graph') -> bool:
        return self.clusters == other.clusters

    def __iter__(self):
        yield from self.clusters

    def __len__(self):
        return len(self.clusters)

    def __str__(self):
        return ';'.join([str(c) for c in self.clusters])

    def __repr__(self):
        return ';'.join([repr(c) for c in self.clusters])

    def __getitem__(self, cluster: 'Cluster') -> 'Cluster':
        # TODO: Get multiple? Graph[c1, c2, c3]? I'd be like np
        return next(iter(self.clusters.intersection(set(cluster))))

    def __contains__(self, cluster: 'Cluster') -> bool:
        return cluster in self.clusters

    def edges(self) -> Set[Set['Cluster']]:
        return set({c, n} for c in self.clusters for n in c.neighbors.keys())

    def subgraphs(self) -> List['Graph']:
        return [Graph(*component) for component in self.components()]

    def components(self) -> Iterable[Set['Cluster']]:
        unvisited = set(self.clusters)
        while unvisited:
            component = self.dft(unvisited.pop())
            unvisited -= component
            yield component

    @lru_cache
    def component(self, cluster: 'Cluster') -> Set['Cluster']:
        return next(filter(lambda component: cluster in component, self.components()))

    def bft(self):
        # TODO
        queue = deque()
        pass

    def dft(self, start: 'Cluster'):
        visited = set()
        stack = [start]
        while stack:
            c = stack.pop()
            if c not in visited:
                visited.add(c)
                stack.extend(c.neighbors.keys())
        return visited


class Manifold:
    # TODO: len(manifold)?

    def __init__(self, data: Data, metric: str, argpoints: Union[Vector, float] = None, **kwargs):
        self.data: Data = data
        self.metric: str = metric

        if argpoints is None:
            self.argpoints = list(range(len(self.data)))
        elif type(argpoints) is list:
            self.argpoints = list(map(int, argpoints))
        elif type(argpoints) is float:
            self.argpoints = list(np.random.choice(len(data), int(len(data) * argpoints), replace=False))
        else:
            raise ValueError(f"Invalid argument to argpoints. {argpoints}")

        self.graphs: List['Graph'] = [Graph(Cluster(self, self.argpoints, ''))]

        self.__dict__.update(**kwargs)
        return

    def __eq__(self, other: 'Manifold') -> bool:
        return all((
            self.metric == other.metric,
            self.graphs[-1] == other.graphs[-1],
        ))

    def __getitem__(self, item):
        return self.graphs[item]

    def __iter__(self):
        yield from self.graphs

    def __str__(self):
        return ';'.join([self.metric, str(self.graphs[-1])])

    def __repr__(self):
        return ';'.join([self.metric, repr(self.graphs[-1])])

    def find_points(self, point: Data, radius: Radius) -> Vector:
        pass

    def find_clusters(self, point: Data, radius: Radius) -> Set['Cluster']:
        pass

    def build(self, *criterion) -> 'Manifold':
        # TODO: This shouldn't rely on graphs[0] being a root element.
        self.graphs = self.graphs[0]
        self.deepen(*criterion)
        return self

    def deepen(self, *criterion) -> 'Manifold':
        while True:
            g = Graph(*[c for C in self.graphs[-1] for c in C.partition(*criterion)])
            if g and g != self.graphs[-1]:
                self.graphs.append(g)
            else:
                break
        return self

    def select(self, name: str) -> Cluster:
        pass

    def dump(self, fp: TextIO) -> None:
        pass

    @staticmethod
    def load(fp: TextIO, data: Data) -> 'Manifold':
        return Manifold()
