import pickle
from collections import deque
from operator import itemgetter
from typing import Set, Dict, TextIO, Iterable

from scipy.spatial.distance import pdist, squareform, cdist

from chess.types import *

SUBSAMPLE_LIMIT = 10
BATCH_SIZE = 10

np.random.seed(42)


class Cluster:
    """ A cluster of points.

    Clusters maintain references to their neighbors (clusters that overlap),
    their children, the manifold to which they belong, and the indices of the
    points they are responsible for.

    You can compare clusters, hash them, partition them, perform tree search,
    prune them, and more. In general, they implement methods that utilize the
    underlying tree structure found across Manifold.graphs.
    """
    # TODO: argpoints -> indices?
    # TODO: Siblings? Maybe a boolean like sibling(other) -> bool?
    # TODO: Consider storing % taken from parent and % of population per cluster
    # TODO: RE above, if we store the percentages we can calculate sparseness

    def __init__(self, manifold: 'Manifold', argpoints: Vector, name: str, **kwargs):
        if len(argpoints) == 0:
            raise ValueError(f"Cluster {name} need argpoints.")

        self.manifold: 'Manifold' = manifold
        self.argpoints: Vector = argpoints
        self.name: str = name

        self.neighbors: Dict['Cluster', float] = dict()  # key is neighbor, value is distance to neighbor
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

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return ','.join([self.name, ';'.join(map(str, self.argpoints))])

    def __len__(self) -> int:
        return len(self.argpoints)

    def __iter__(self) -> Vector:
        for i in range(0, len(self), BATCH_SIZE):
            yield self.argpoints[i:i + BATCH_SIZE]

    def __getitem__(self, item: int) -> Data:
        return self.manifold.data[self.argpoints[item]]

    def __contains__(self, point: Data) -> bool:
        return self.overlaps(point=point, radius=0.)

    @property
    def metric(self) -> str:
        """ The metric used in the manifold. """
        return self.manifold.metric

    @property
    def depth(self) -> int:
        """ The depth at which the cluster exists. """
        return len(self.name)

    @property
    def data(self) -> Data:
        """ Iterates over the data that the cluster owns by BATCH_SIZE. """
        for i in range(0, len(self), BATCH_SIZE):
            yield self.manifold.data[self.argpoints[i:i + BATCH_SIZE]]

    @property
    def points(self) -> Data:
        """ Same as self.data."""
        yield from self.data

    @property
    def samples(self) -> Data:
        """ Returns the samples from the cluster.

        Samples are used in computing approximate centers and poles.
        """
        return self.manifold.data[self.argsamples]

    @property
    def argsamples(self) -> Vector:
        """ Indices used to retrieve samples.

        Ensures that there are at least 2 different points in samples,
        otherwise returns a single sample that represents the entire cluster.
        AKA, if len(argsamples) == 1, the cluster contains only duplicates.
        """
        if '_argsamples' not in self.__dict__:
            if len(self) <= SUBSAMPLE_LIMIT:
                n = len(self.argpoints)
                indices = self.argpoints
            else:
                n = int(np.sqrt(len(self)))
                indices = np.random.choice(self.argpoints, n, replace=False)

            # Handle Duplicates.
            if pdist(self.manifold.data[indices], self.metric).max(initial=0.) == 0.:
                indices = np.unique(self.manifold.data[self.argpoints], return_index=True, axis=0)[1]
                if len(indices) > n:
                    indices = np.random.choice(indices, n, replace=False)
                indices = [self.argpoints[i] for i in indices]

            # Cache it.
            self.__dict__['_argsamples'] = indices
        return self.__dict__['_argsamples']

    @property
    def nsamples(self) -> int:
        """ The number of samples for the cluster. """
        return len(self.argsamples)

    @property
    def center(self) -> Data:
        """ The centroid of the cluster. """
        return self.manifold.data[self.argcenter]

    @property
    def argcenter(self) -> int:
        """ The index used to retrieve the center. """
        if '_argcenter' not in self.__dict__:
            self.__dict__['_argcenter'] = self.argsamples[int(np.argmin(squareform(pdist(self.samples)).sum(axis=1)))]
        return self.__dict__['_argcenter']

    @property
    def radius(self) -> Radius:
        """ The radius of the cluster.

        Computed as distance from center to farthest point.
        """
        if '_min_radius' in self.__dict__:
            return self.__dict__['_min_radius']
        elif '_radius' not in self.__dict__:
            _ = self.argradius
        return self.__dict__['_radius']

    @property
    def argradius(self) -> int:
        """ The index used to retrieve the point which is farthest from the center.
        """
        if ('_argradius' not in self.__dict__) or ('_radius' not in self.__dict__):
            def argmax_max(b):
                distances = self.distance(self.manifold.data[b])
                argmax = np.argmax(distances)
                return b[argmax], distances[argmax]

            # noinspection PyTypeChecker
            argradii_radii = [argmax_max(batch) for batch in self]
            self.__dict__['_argradius'], self.__dict__['_radius'] = max(argradii_radii, key=itemgetter(1))
        return self.__dict__['_argradius']

    @property
    def local_fractal_dimension(self) -> float:
        """ The local fractal dimension of the cluster. """
        if self.nsamples == 1:
            return Radius(0.)
        count = [d <= (self.radius / 2)
                 for batch in self
                 for d in self.distance(self.manifold.data[batch])]
        count = np.sum(count)
        return count if count == 0. else np.log2(len(self.argpoints) / count)

    def clear_cache(self) -> None:
        """ Clears the cache for the cluster. """
        for prop in ['_argsamples', '_argcenter', '_argradius', '_radius']:
            try:
                del self.__dict__[prop]
            except KeyError:
                pass

    def tree_search(self, point: Data, radius: Radius, depth: int) -> List['Cluster']:
        """ Searches down the tree for clusters that overlap point with radius at depth.
        """
        if depth == -1:
            depth = len(self.manifold.graphs)
        if depth < self.depth:
            raise ValueError('depth must not be less than cluster.depth')
        results: List[Cluster] = list()
        if self.overlaps(point, radius):
            # results ONLY contains clusters that have overlap with point
            results.append(self)
            for d in range(self.depth, depth):
                children: List[Cluster] = [c for candidate in results for c in candidate.children]
                if len(children) == 0:
                    break
                centers = np.asarray([c.center for c in children])
                distances = cdist(np.expand_dims(point, 0), centers, self.metric)[0]
                radii = [radius + c.radius for c in children]
                results = [c for c, d, r in zip(children, distances, radii) if d <= r]
                if len(results) == 0:
                    break
        assert all((depth == n.depth) for n in results)
        return results

    def prune(self) -> None:
        """ Removes all references to descendents. """
        if self.children:
            [c.neighbors.remove(c) for c in self.children]
            [c.prune() for c in self.children]
            self.children = set()
        return

    def partition(self, *criterion) -> Iterable['Cluster']:
        """ Partitions the cluster into 1-2 children.

        2 children are produced if the cluster can be split,
        otherwise 1 child is produced.
        """
        if not all((
                len(self.argpoints) > 1,
                len(self.argsamples) > 1,
                *(c(self) for c in criterion)
        )):
            self.children = {Cluster(self.manifold, self.argpoints, self.name + '0')}
            return self.children

        farthest = self.argsamples[int(np.argmax(cdist(
            np.expand_dims(self.manifold.data[self.argradius], 0),
            self.samples,
            self.metric
        )[0]))]
        poles = np.stack([
            self.manifold.data[self.argradius],
            self.manifold.data[farthest]
        ])

        p1_idx, p2_idx = list(), list()
        for batch in self:  # TODO: Comprehension? np.concatenate((batch, distances))
            distances = cdist(poles, self.manifold.data[batch], self.metric)
            # noinspection PyTypeChecker
            [(p1_idx if p1 < p2 else p2_idx).append(i) for i, p1, p2 in zip(batch, distances[0], distances[1])]

        self.children = {
            Cluster(self.manifold, p1_idx, self.name + '1'),
            Cluster(self.manifold, p2_idx, self.name + '2'),
        }
        return self.children

    def update_neighbors(self) -> Dict['Cluster', Radius]:
        """ Find neighbors, update them, return the set. """
        neighbors = list(self.manifold.find_clusters(self.center, self.radius, self.depth) - {self})
        if len(neighbors) == 0:
            self.neighbors = dict()
        else:
            assert all((self.depth == n.depth) for n in neighbors)
            distances = self.distance(np.asarray([n.center for n in neighbors]))
            self.neighbors = {n: d for n, d in zip(neighbors, distances)}
            [n.neighbors.update({self: d}) for n, d in self.neighbors.items()]
        return self.neighbors

    def distance(self, points: Data) -> np.ndarray:
        """ Returns the distance from self.center to every point in points. """
        return cdist(np.expand_dims(self.center, 0), points, self.metric)[0]

    def overlaps(self, point: Data, radius: Radius) -> bool:
        """ Checks if point is within radius + self.radius of cluster. """
        return self.distance(np.expand_dims(point, axis=0))[0] <= (self.radius + radius)

    def json(self):
        return {
            'name': self.name,
            'argpoints': self.argpoints,
        }

    def dump(self, fp) -> None:
        pass

    @staticmethod
    def load(fp):
        pass


class Graph:
    """ Graph comprised of clusters.

    Graphs hold sets of clusters, which are the vertices.
    The graph class is responsible for handling operations that occur
    solely within a layer of Manifold.graphs.
    """
    def __init__(self, *clusters):
        assert all([c.depth == clusters[0].depth for c in clusters[1:]])
        self.clusters = set(clusters)
        return

    def __eq__(self, other: 'Graph') -> bool:
        return self.clusters == other.clusters

    def __iter__(self) -> Iterable[Cluster]:
        yield from self.clusters

    def __len__(self) -> int:
        return len(self.clusters)

    def __str__(self) -> str:
        return ', '.join(sorted([str(c) for c in self.clusters]))

    def __repr__(self) -> str:
        return '\t'.join(sorted([repr(c) for c in self.clusters]))

    def __contains__(self, cluster: 'Cluster') -> bool:
        return cluster in self.clusters

    @property
    def edges(self) -> Set[Set['Cluster']]:
        """ Returns all edges within the graph.
        """
        if '_edges' not in self.__dict__:
            self.__dict__['_edges'] = set({c, n} for c in self.clusters for n in c.neighbors.keys())
        return self.__dict__['_edges']

    @property
    def subgraphs(self) -> List['Graph']:
        """ Returns all subgraphs within the graph.
        """
        if '_subgraphs' not in self.__dict__:
            self.__dict__['_subgraphs'] = [Graph(*component) for component in self.components]
        return self.__dict__['_subgraphs']

    @property
    def components(self) -> List[Set['Cluster']]:
        """ Returns all components within the graph.
        """
        # TODO: Isn't this the same thing as subgraphs?
        if '_components' not in self.__dict__:
            unvisited = set(self.clusters)
            self.__dict__['_components'] = list()
            while unvisited:
                component = self.bft(unvisited.pop())
                unvisited -= component
                self.__dict__['_components'].append(component)
        return self.__dict__['_components']

    def clear_cache(self) -> None:
        """ Clears the cache of the graph. """
        for prop in ['_components', '_subgraphs', '_edges']:
            try:
                del self.__dict__[prop]
            except KeyError:
                pass

    def component(self, cluster: 'Cluster') -> Set['Cluster']:
        """ Returns the component to which cluster belongs. """
        return next(filter(lambda component: cluster in component, self.components))

    @staticmethod
    def bft(start: 'Cluster'):
        """ Breadth-First Search starting at start. """
        visited = set()
        queue = deque([start])
        while queue:
            c = queue.popleft()
            if c not in visited:
                visited.add(c)
                # [queue.append(neighbor) for neighbor in c.neighbors.keys()]
                neighbors = c.manifold.find_clusters(c.center, c.radius, c.depth) - {c}
                [queue.append(neighbor) for neighbor in neighbors]
        return visited

    @staticmethod
    def dft(start: 'Cluster'):
        """ Depth-First Search starting at start. """
        visited = set()
        stack = [start]
        while stack:
            c = stack.pop()
            if c not in visited:
                visited.add(c)
                stack.extend(c.neighbors.keys())
        return visited


class Manifold:
    """ Manifold of varying detail.

    The manifold class' main job is to organize the underlying graphs of
    various depths. With this stack of graphs, manifold provides utilities
    for finding clusters and points that overlap with a given query point
    with a radius, along with providing the ability to reset the build
    of the list of graphs, and iteratively deepening it.
    """

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

    def __getitem__(self, depth: int) -> 'Graph':
        return self.graphs[depth]

    def __iter__(self) -> Iterable[Graph]:
        yield from self.graphs

    def __str__(self) -> str:
        return '\t'.join([self.metric, str(self.graphs[-1])])

    def __repr__(self) -> str:
        return '\n'.join([self.metric, repr(self.graphs[-1])])

    def find_points(self, point: Data, radius: Radius) -> Vector:
        """ Returns all indices of points that are within radius of point. """
        candidates = [p for c in self.find_clusters(point, radius, len(self.graphs)) for p in c.argpoints]
        results: Vector = list()
        point = np.expand_dims(point, axis=0)
        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i:i + BATCH_SIZE]
            distances = cdist(point, self.data[batch], self.metric)[0]
            results.extend([p for p, d in zip(batch, distances) if d <= radius])
        return results

    def find_clusters(self, point: Data, radius: Radius, depth: int) -> Set['Cluster']:
        """ Returns all clusters within radius of point at depth. """
        return {r for c in self.graphs[0] for r in c.tree_search(point, radius, depth)}

    def build(self, *criterion) -> 'Manifold':
        """ Rebuilds the stack of graphs. """
        self.graphs = [Graph(Cluster(self, self.argpoints, ''))]
        self.deepen(*criterion)
        return self

    def deepen(self, *criterion) -> 'Manifold':
        """ Iteratively deepens the stack of graphs whilst checking criterion. """
        while True:
            clusters = [child for cluster in self.graphs[-1] for child in cluster.partition(*criterion)]
            if len(self.graphs[-1]) < len(clusters):
                g = Graph(*clusters)
                self.graphs.append(g)
                [c.update_neighbors() for c in g]
            else:
                break
        return self

    def select(self, name: str) -> Cluster:
        """ Returns the cluster with the given name. """
        # TODO: Support clipping of layers from self.graphs
        return next(filter(lambda c: c.name == name, self.graphs[len(name)]))

    def dump(self, fp: TextIO) -> None:
        pickle.dump({
            'metric': self.metric,
            'graphs': [self.graphs[-1]]},
            fp
        )
        return

    @staticmethod
    def load(fp: TextIO, data: Data) -> 'Manifold':
        m = Manifold(data, **pickle.load(fp))
        # TODO: Rebuild upper layers.
        return m
