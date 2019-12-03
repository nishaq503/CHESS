from typing import Set, Dict, TextIO, Iterable

from scipy.spatial.distance import pdist

from chess.types import *

SUBSAMPLE_LIMIT = 10


class Cluster:
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

    @property
    def metric(self):
        return self.manifold.metric

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
    def argcenter(self):
        pass

    @property
    def center(self):
        pass

    @property
    def radius(self):
        pass

    @property
    def local_fractal_dimension(self):
        pass

    def clear_cache(self):
        for prop in ['_argsamples']:
            try:
                del self.__dict__[prop]
            except KeyError:
                pass

    def tree_search(self, point: Data, radius: Radius) -> Set['Cluster']:
        """ Searches down the tree. """
        pass

    def prune(self) -> None:
        """ Removes children. """
        pass

    def partition(self, *criterion) -> Iterable['Cluster']:
        pass

    def update_neighbors(self) -> Set['Cluster']:
        """ Find neighbors, update them, return the set. """
        pass

    def overlaps(self, other: 'Cluster') -> bool:
        pass

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

    def subgraphs(self) -> List['Graph']:
        pass

    def components(self) -> Set[Set['Cluster']]:
        pass

    def component(self, cluster: 'Cluster') -> Set['Cluster']:
        pass

    def bft(self):
        pass

    def dft(self):
        pass


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
        return self

    def deepen(self, *criterion) -> 'Manifold':
        return self

    def select(self, name: str) -> Cluster:
        pass

    def dump(self, fp: TextIO) -> None:
        pass

    @staticmethod
    def load(fp: TextIO, data: Data) -> 'Manifold':
        return Manifold()
