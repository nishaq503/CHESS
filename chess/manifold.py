import json
import pickle
from collections import deque
from operator import itemgetter
from typing import Set, Dict, Iterable, Deque, TextIO, BinaryIO

from scipy.spatial.distance import pdist, cdist

from chess.types import *

SUBSAMPLE_LIMIT = 10
BATCH_SIZE = 10
MIN_RADIUS = 0.


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
        return self.name == other.name and all((m == o for m, o in zip(sorted(self.argpoints), sorted(other.argpoints))))

    def __hash__(self):
        # TODO: Investigate int base k conversions.
        # if 'loading' in self.manifold.__dict__
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
        return self.manifold.data[self.argpoints[item]]  # TODO: Cover

    def __contains__(self, point: Data) -> bool:  # TODO: Cover
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
    def data(self) -> Data:  # TODO: Cover
        """ Iterates over the data that the cluster owns by BATCH_SIZE. """
        for i in range(0, len(self), BATCH_SIZE):
            yield self.manifold.data[self.argpoints[i:i + BATCH_SIZE]]

    @property
    def points(self) -> Data:  # TODO: Cover
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
                    indices = np.random.choice(indices, n, replace=False)  # TODO: Cover
                indices = [self.argpoints[i] for i in indices]

            # Cache it.
            self.__dict__['_argsamples'] = indices
        return self.__dict__['_argsamples']

    @property
    def nsamples(self) -> int:
        """ The number of samples for the cluster. """
        return len(self.argsamples)

    @property
    def center(self) -> Data:  # TODO: slow
        """ The centroid of the cluster. """
        return self.manifold.data[self.argcenter]

    @property
    def argcenter(self) -> int:  # TODO: slow
        """ The index used to retrieve the center. """
        if '_argcenter' not in self.__dict__:
            self.__dict__['_argcenter'] = self.argsamples[int(np.argmin(cdist(self.samples, self.samples, self.metric).sum(axis=1)))]
        return self.__dict__['_argcenter']

    @property
    def radius(self) -> Radius:  # TODO: Slow
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
            self.__dict__['_radius'] = max(self.__dict__['_radius'], MIN_RADIUS)
        return self.__dict__['_argradius']

    @property
    def local_fractal_dimension(self) -> float:  # TODO: Cover
        """ The local fractal dimension of the cluster. """
        if '_local_fractal_dimension' not in self.__dict__:
            if self.nsamples == 1:
                return 0.
            count = [d <= (self.radius / 2)
                     for batch in self
                     for d in self.distance(self.manifold.data[batch])]
            count = np.sum(count)
            self.__dict__['_local_fractal_dimension'] = count if count == 0. else np.log2(len(self.argpoints) / count)
        return self.__dict__['_local_fractal_dimension']

    def clear_cache(self) -> None:  # TODO: Cover
        """ Clears the cache for the cluster. """
        for prop in ['_argsamples', '_argcenter', '_argradius', '_radius', '_local_fractal_dimension']:
            try:
                del self.__dict__[prop]
            except KeyError:
                pass

    def tree_search(self, point: Data, radius: Radius, depth: int, mode: str) -> List['Cluster']:
        """ Searches down the tree for clusters that overlap point with radius at depth.
        """
        results: List[Cluster] = list()
        if mode == 'iterative':
            if depth == -1:
                depth = len(self.manifold.graphs)
            if depth < self.depth:
                raise ValueError('depth must not be less than cluster.depth')
            if self.overlaps(point, radius):
                # results ONLY contains clusters that have overlap with point
                results.append(self)
                for d in range(self.depth, depth):
                    children: List[Cluster] = [c for candidate in results for c in candidate.children]
                    if len(children) == 0:
                        break  # TODO: Cover. Is this even possible?
                    centers = np.asarray([c.center for c in children])
                    distances = cdist(np.expand_dims(point, 0), centers, self.metric)[0]
                    radii = [radius + c.radius for c in children]
                    results = [c for c, d, r in zip(children, distances, radii) if d <= r]
                    if len(results) == 0:
                        break  # TODO: Cover
                assert depth == results[0].depth, (depth, results[0].depth)
                assert all((depth == r.depth for r in results))
        elif mode == 'recursive':
            assert self.overlaps(point, radius)
            if depth == -1:
                depth = len(self.manifold.graphs)  # TODO: Cover
            if self.radius <= radius or len(self.children) < 2:
                results.append(self)  # TODO: Cover
            elif self.depth < depth:
                if len(self.children) < 1:
                    results.append(self)  # TODO: Cover Is this even possible?
                else:
                    [results.append(c) for c in self.children if c.overlaps(point, radius)]
        elif mode == 'dfs':
            stack: List[Cluster] = [self]
            while len(stack) > 0:
                current: Cluster = stack.pop()
                if current.overlaps(point, radius):
                    if current.depth < depth:
                        if len(current.children) < 2:
                            results.append(current)
                        else:
                            stack.extend(current.children)
                    else:
                        results.append(current)  # TODO: Cover
        elif mode == 'bfs':
            queue: Deque[Cluster] = deque([self])
            while len(queue) > 0:
                current: Cluster = queue.popleft()
                if current.overlaps(point, radius):
                    if current.depth < depth:
                        if len(current.children) < 2:
                            results.append(current)
                        else:
                            [queue.append(c) for c in current.children]
                    else:
                        results.append(current)  # TODO: Cover
        else:
            raise ValueError(f'mode must be one of iterative, recursive, dfs, or bfs. got {mode} instead.')  # TODO: Cover
        return results

    def prune(self) -> None:  # TODO: Cover
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
            self.children = {
                Cluster(
                    self.manifold,
                    self.argpoints,
                    self.name + '0',
                    _argsamples=self.argsamples,
                    _argcenter=self.argcenter,
                    _argradius=self.argradius,
                    _radius=self.radius,
                    )
            }
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

    def add_edge(self, other: 'Cluster', propagate: bool):
        if self.depth != other.depth:
            raise ValueError(f'Cannot add an edge between two clusters at different depths.')  # TODO: Cover
        if propagate:
            lineages = [(self.name[:i], other.name[:i]) for i in range(self.depth + 1) if self.name[:i] != other.name[:i]]
            clusters = [(self.manifold.select(l1), self.manifold.select(l2)) for l1, l2 in lineages]
            centers = np.stack([c.center for c, _ in clusters], axis=0), np.stack([c.center for _, c in clusters], axis=0)
            assert centers[0].shape == centers[1].shape, (centers[0].shape, centers[1].shape)
            if len(centers[0].shape) == 1:
                centers = np.expand_dims(centers[0], axis=0), np.expand_dims(centers[1], axis=0)  # TODO: Cover
            distances = list(np.diag(cdist(centers[0], centers[1], self.metric)))
            [(c1.neighbors.update({c2: d}), c2.neighbors.update({c1: d})) for (c1, c2), d in zip(clusters, distances)]
        else:  # TODO: Cover
            distance = self.distance(np.expand_dims(other.center, axis=0))[0]
            self.neighbors.update({other: distance}), other.neighbors.update({self: distance})
        return

    def update_neighbors(self, propagate: bool = True) -> Dict['Cluster', Radius]:
        """ Find neighbors, update them, return the set. """
        def _update():
            if len(neighbors) != 0:
                centers = np.stack([c.center for c in neighbors], axis=0)
                if len(centers.shape) == 0:
                    centers = np.expand_dims(centers, axis=0)  # TODO: Cover
                distances = list(self.distance(centers))
                radii = [self.radius + c.radius for c in neighbors]  # TODO: low
                [self.add_edge(c, propagate) for c, d, r in zip(neighbors, distances, radii) if d <= r]
            return

        neighbors = [n for n in list(self.manifold.find_clusters(self.center, self.radius, self.depth) - {self})
                     if self not in set(n.neighbors.keys()) and n not in set(self.neighbors.keys())]
        _update()
        if propagate:
            neighbors = [c for c in self.manifold.graphs[self.depth]
                         if (self.name != c.name) and (self not in c.neighbors) and (c not in self.neighbors)]  # TODO: slow
            _update()
        return self.neighbors

    def distance(self, points: Data) -> np.ndarray:
        """ Returns the distance from self.center to every point in points. """
        return cdist(np.expand_dims(self.center, 0), points, self.metric)[0]

    def overlaps(self, point: Data, radius: Radius) -> bool:
        """ Checks if point is within radius + self.radius of cluster. """
        return self.distance(np.expand_dims(point, axis=0))[0] <= (self.radius + radius)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'argpoints': list(map(int, self.argpoints)),
            'argsamples': list(map(int, self.argsamples)),
            'argcenter': int(self.argcenter),
            'argradius': int(self.argradius),
            'radius': float(self.radius),
            'local_fractal_dimension': float(self.local_fractal_dimension),
            'children': [c.to_dict() for c in self.children],
            'neighbors': {n.name: d for n, d in self.neighbors.items()},
        }

    @staticmethod
    def from_dict(manifold: 'Manifold', loaded_dict: Dict):
        cluster = Cluster(
            manifold=manifold,
            argpoints=loaded_dict['argpoints'],
            name=loaded_dict['name'],
            _argsamples=loaded_dict['argsamples'],
            _argcenter=loaded_dict['argcenter'],
            _argradius=loaded_dict['argradius'],
            _radius=loaded_dict['radius'],
            _local_fractal_dimension=loaded_dict['local_fractal_dimension'],
            neighbors=loaded_dict['neighbors']
        )
        cluster.children = [Cluster.from_dict(manifold, ld) for ld in loaded_dict['children']]
        return cluster


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
        my_clusters = sorted([c.name for c in self.clusters])
        other_clusters = sorted([c.name for c in other.clusters])
        return all((m == o for m, o in zip(my_clusters, other_clusters)))

    def __iter__(self) -> Iterable[Cluster]:
        yield from self.clusters

    def __len__(self) -> int:
        return len(self.clusters)

    def __str__(self) -> str:
        return ', '.join(sorted([str(c) for c in self.clusters]))

    def __repr__(self) -> str:
        return '\t'.join(sorted([repr(c) for c in self.clusters]))

    def __contains__(self, cluster: 'Cluster') -> bool:  # TODO: Cover
        return cluster in self.clusters

    @property
    def edges(self) -> Set[Set['Cluster']]:  # TODO: Cover
        """ Returns all edges within the graph.
        """
        if '_edges' not in self.__dict__:
            self.__dict__['_edges'] = set({c, n} for c in self.clusters for n in c.neighbors.keys())
        return self.__dict__['_edges']

    @property
    def subgraphs(self) -> List['Graph']:  # TODO: Cover
        """ Returns all subgraphs within the graph.
        """
        if '_subgraphs' not in self.__dict__:
            self.__dict__['_subgraphs'] = [Graph(*component) for component in self.components]
        return self.__dict__['_subgraphs']

    @property
    def components(self) -> List[Set['Cluster']]:  # TODO: Cover
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

    def clear_cache(self) -> None:  # TODO: Cover
        """ Clears the cache of the graph. """
        for prop in ['_components', '_subgraphs', '_edges']:
            try:
                del self.__dict__[prop]
            except KeyError:
                pass

    def component(self, cluster: 'Cluster') -> Set['Cluster']:  # TODO: Cover
        """ Returns the component to which cluster belongs. """
        return next(filter(lambda component: cluster in component, self.components))

    @staticmethod
    def bft(start: 'Cluster'):  # TODO: Cover
        """ Breadth-First Search starting at start. """
        visited = set()
        queue = deque([start])
        while queue:
            c = queue.popleft()
            if c not in visited:
                visited.add(c)
                [queue.append(neighbor) for neighbor in c.neighbors.keys()]
        return visited

    @staticmethod
    def dft(start: 'Cluster'):  # TODO: Cover
        """ Depth-First Search starting at start. """
        visited = set()
        stack: List[Cluster] = [start]
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
            raise ValueError(f"Invalid argument to argpoints. {argpoints}")  # TODO: Cover

        self.__dict__.update(**kwargs)
        if 'loading' in self.__dict__ and self.__dict__['loading'] is True:
            self.graphs: List['Graph'] = list()
        else:
            self.graphs: List['Graph'] = [Graph(Cluster(self, self.argpoints, ''))]
        return

    def __eq__(self, other: 'Manifold') -> bool:
        return self.metric == other.metric and self.graphs[-1] == other.graphs[-1]

    def __getitem__(self, depth: int) -> 'Graph':
        return self.graphs[depth]

    def __iter__(self) -> Iterable[Graph]:
        yield from self.graphs

    def __str__(self) -> str:
        return '\t'.join([self.metric, str(self.graphs[-1])])

    def __repr__(self) -> str:
        return '\n'.join([self.metric, repr(self.graphs[-1])])

    def find_points(self, point: Data, radius: Radius, mode: str = 'iterative') -> Vector:
        """ Returns all indices of points that are within radius of point. """
        candidates = [p for c in self.find_clusters(point, radius, len(self.graphs), mode) for p in c.argpoints]
        results: Vector = list()
        point = np.expand_dims(point, axis=0)
        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i:i + BATCH_SIZE]
            distances = cdist(point, self.data[batch], self.metric)[0]
            results.extend([p for p, d in zip(batch, distances) if d <= radius])
        return results

    def find_clusters(self, point: Data, radius: Radius, depth: int, mode: str = 'iterative') -> Set['Cluster']:
        """ Returns all clusters within radius of point at depth. """
        return {r for c in self.graphs[0] for r in c.tree_search(point, radius, depth, mode)}

    def build(self, *criterion) -> 'Manifold':
        """ Rebuilds the stack of graphs. """
        self.graphs = [Graph(Cluster(self, self.argpoints, ''))]
        self.deepen(*criterion)
        return self

    def deepen(self, *criterion) -> 'Manifold':
        """ Iteratively deepens the stack of graphs whilst checking criterion. """
        while True:
            clusters = [child for cluster in self.graphs[-1] for child in cluster.partition(*criterion)]
            if len(clusters) != len(self.graphs[-1]):
                g = Graph(*clusters)
                self.graphs.append(g)
                if 'propagate' not in self.__dict__:
                    self.__dict__['propagate'] = False
                if 'calculate_neighbors' in self.__dict__ and self.__dict__['calculate_neighbors'] is False:
                    for c in clusters:
                        c.neighbors = dict()
                    continue
                else:
                    [c.update_neighbors(propagate=self.__dict__['propagate']) for c in clusters]
            else:
                break
        return self

    def select(self, name: str) -> Cluster:
        """ Returns the cluster with the given name. """
        # TODO: Support multiple roots
        # TODO: Support clipping of layers from self.graphs
        cluster: Cluster = next(iter(self.graphs[0]))
        for depth in range(len(name) + 1):
            partial_name = name[:depth]
            for child in cluster.children:
                if child.name == partial_name:
                    cluster = child
                    break
        assert name == cluster.name, f'wanted {name} but got {cluster.name}.'
        return cluster

    def to_dict(self) -> Dict:
        root_dict = self.select('').to_dict()
        return {
            'metric': self.metric,
            'argpoints': list(map(int, self.argpoints)),
            'root': root_dict,
        }

    def json_dump(self, fp: TextIO):  # TODO:
        my_dict = self.to_dict()
        json.dump(my_dict, fp)
        return

    @staticmethod
    def from_dict(loaded_dict, data: Data, **kwargs) -> 'Manifold':
        manifold: 'Manifold' = Manifold(
            data=data,
            metric=loaded_dict['metric'],
            argpoints=loaded_dict['argpoints'],
            **kwargs,
        )
        root: 'Cluster' = Cluster.from_dict(manifold, loaded_dict['root'])
        graphs: List['Graph'] = [Graph(root)]
        while True:
            graph: Graph = graphs[-1]
            clusters: List['Cluster'] = [cluster for cluster in graph if len(cluster.children) > 0]
            if len(clusters) == 0:
                if len(graphs) > 1:
                    graphs.pop()
                break
            children = [child for cluster in clusters for child in cluster.children]
            graphs.append(Graph(*children))

        manifold.graphs = graphs

        for graph in manifold.graphs:
            for cluster in graph.clusters:
                cluster.neighbors = {manifold.select(n): d for n, d in cluster.__dict__['neighbors'].items()}
                del cluster.__dict__['neighbors']
        return manifold

    @staticmethod
    def json_load(fp: TextIO, data: Data) -> 'Manifold':  # TODO:
        loaded_dict = json.load(fp)
        return Manifold.from_dict(loaded_dict, data)

    def pickle_dump(self, fp: BinaryIO):
        my_dict = self.to_dict()
        pickle.dump(my_dict, fp)
        return

    @staticmethod
    def pickle_load(fp: BinaryIO, data: Data) -> 'Manifold':
        loaded_dict = pickle.load(fp)
        return Manifold.from_dict(loaded_dict, data)
