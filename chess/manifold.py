""" Clustered Hierarchical Entropy-scaling Manifold Mapping.

# TODO: https://docs.python.org/3/whatsnew/3.8.html#f-strings-support-for-self-documenting-expressions-and-debugging
"""
import logging
import pickle
from collections import deque
from operator import itemgetter
from queue import Queue
from threading import Thread
from typing import Set, Dict, Iterable, TextIO

from scipy.spatial.distance import pdist, cdist

from chess.types import *

SUBSAMPLE_LIMIT = 100
BATCH_SIZE = 10_000
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s:%(levelname)s:%(name)s:%(module)s.%(funcName)s:%(message)s"
)


class Cluster:
    """ A cluster of points.

    Clusters maintain references to their neighbors (clusters that overlap),
    their children, the manifold to which they belong, and the indices of the
    points they are responsible for.

    You can compare clusters, hash them, partition them, perform tree search,
    prune them, and more. In general, they implement methods that utilize the
    underlying tree structure found across Manifold.graphs.
    """

    def __init__(self, manifold: 'Manifold', argpoints: Vector, name: str, **kwargs):
        logging.debug(f"Cluster(name={name}, argpoints={argpoints})")
        self.manifold: 'Manifold' = manifold
        self.argpoints: Vector = argpoints
        self.name: str = name

        self.neighbors: Dict['Cluster', float] = dict()  # key is neighbor, value is distance to neighbor
        self.children: Set['Cluster'] = set()

        self.__dict__.update(**kwargs)

        # This is used during Cluster.from_json().
        if not argpoints and self.children:
            self.argpoints = [p for child in self.children for p in child.argpoints]
        elif not argpoints:
            raise ValueError(f"Cluster {name} need argpoints.")
        return

    def __eq__(self, other: 'Cluster') -> bool:
        return all((
            self.name == other.name,
            set(self.argpoints) == set(other.argpoints),
        ))

    def __hash__(self):
        return hash(self.name)

    def __str__(self) -> str:
        return self.name or 'root'

    def __repr__(self) -> str:
        return ','.join([self.name, ';'.join(map(str, self.argpoints))])

    def __len__(self) -> int:
        """ Returns cardinality of the set of points.
        TODO: Consider deprecating __len__ and providing Cluster().cardinality
        """
        return len(self.argpoints)

    def __iter__(self) -> Vector:
        # Iterates in batches, instead of by element.
        for i in range(0, len(self), BATCH_SIZE):
            yield self.argpoints[i:i + BATCH_SIZE]

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
    def points(self) -> Data:
        """ Same as self.data."""
        for i in range(0, len(self), BATCH_SIZE):
            yield self.manifold.data[self.argpoints[i:i + BATCH_SIZE]]

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
            logging.debug(f"building cache for {self}")
            if len(self) <= SUBSAMPLE_LIMIT:
                n = len(self.argpoints)
                indices = self.argpoints
            else:
                n = int(np.sqrt(len(self)))
                indices = list(np.random.choice(self.argpoints, n, replace=False))

            # Handle Duplicates.
            if pdist(self.manifold.data[indices], self.metric).max(initial=0.) == 0.:
                indices = np.unique(self.manifold.data[self.argpoints], return_index=True, axis=0)[1]
                indices = [self.argpoints[i] for i in indices][:n]

            # Cache it.
            self.__dict__['_argsamples'] = indices
        return self.__dict__['_argsamples']

    @property
    def nsamples(self) -> int:
        """ The number of samples for the cluster. """
        return len(self.argsamples)

    @property
    def centroid(self) -> Data:
        """ The centroid of the cluster. (Geometric Mean) """
        return np.average(self.samples, axis=0)

    @property
    def medoid(self) -> Data:
        """ The medoid of the cluster. (Geometric Median) """
        return self.manifold.data[self.argmedoid]

    @property
    def argmedoid(self) -> int:
        """ The index used to retrieve the medoid. """
        if '_argmedoid' not in self.__dict__:
            logging.debug(f"building cache for {self}")
            _argmedoid = np.argmin(cdist(self.samples, self.samples, self.metric).sum(axis=1))
            self.__dict__['_argmedoid'] = self.argsamples[int(_argmedoid)]
        return self.__dict__['_argmedoid']

    @property
    def radius(self) -> Radius:
        """ The radius of the cluster.

        Computed as distance from medoid to farthest point.
        """
        if '_min_radius' in self.__dict__:
            logging.debug(f'taking min_radius from {self}')
            return self.__dict__['_min_radius']
        elif '_radius' not in self.__dict__:
            logging.debug(f'building cache for {self}')
            _ = self.argradius
        return self.__dict__['_radius']

    @property
    def argradius(self) -> int:
        """ The index used to retrieve the point which is farthest from the medoid.
        """
        if ('_argradius' not in self.__dict__) or ('_radius' not in self.__dict__):
            logging.debug(f'building cache for {self}')

            def argmax_max(b):
                distances = self.distance(self.manifold.data[b])
                argmax = int(np.argmax(distances))
                return b[argmax], distances[argmax]

            argradii_radii = [argmax_max(batch) for batch in iter(self)]
            _argradius, _radius = max(argradii_radii, key=itemgetter(1))
            self.__dict__['_argradius'], self.__dict__['_radius'] = int(_argradius), float(_radius)
        return self.__dict__['_argradius']

    @property
    def local_fractal_dimension(self) -> float:
        """ The local fractal dimension of the cluster. """
        if '_local_fractal_dimension' not in self.__dict__:
            logging.debug(f'building cache for {self}')
            if self.nsamples == 1:
                return 0.
            count = [d <= (self.radius / 2)
                     for batch in self
                     for d in self.distance(self.manifold.data[batch])]
            count = np.sum(count)
            self.__dict__['_local_fractal_dimension'] = count if count == 0. else np.log2(len(self.argpoints) / count)
        return self.__dict__['_local_fractal_dimension']

    def clear_cache(self) -> None:
        """ Clears the cache for the cluster. """
        logging.debug(f'clearing cache for {self}')
        for prop in ['_argsamples', '_argmedoid', '_argradius', '_radius', '_local_fractal_dimension']:
            try:
                del self.__dict__[prop]
            except KeyError:
                pass

    def update_neighbors(self, threshold: int) -> Dict['Cluster', float]:
        """ Finds all neighbors of self.

        :param threshold: how far to look back in the tree to find potential neighbors.
        :return: dictionary of neighbor cluster and distance to that cluster.
        """
        logging.debug(self.name)
        root: Cluster = self.manifold.select('')

        potential_neighbors: Dict[Cluster, float] = {root: 0}
        for depth in range(1, self.depth + 1):
            potential_neighbors = {child: t for c, t in potential_neighbors.items() for child in c.children}
            logging.debug(f'depth: {depth}, number of potential_neighbors: {len(potential_neighbors.keys())}\n')
            centers = np.asarray([c.medoid for c in potential_neighbors.keys()])
            distances = cdist(np.expand_dims(self.medoid, 0), centers, self.metric)[0]
            radii = [self.radius + 2 * c.radius for c in potential_neighbors.keys()]
            potential_neighbors = {c: (0 if d <= r else t + 1)
                                   for (c, t), d, r in zip(potential_neighbors.items(), distances, radii)}
            potential_neighbors = {c: t for c, t in potential_neighbors.items() if t < threshold}
            logging.debug(f'depth: {depth}, number of potential_neighbors: {len(potential_neighbors.keys())}\n')

        assert self in potential_neighbors, f'{self.name} not in potential neighbors'
        del potential_neighbors[self]

        potential_neighbors: List[Cluster] = [c for c in potential_neighbors.keys()]
        centers = np.asarray([c.medoid for c in potential_neighbors])
        if centers.shape[0] < 1:
            return self.neighbors
        distances = cdist(np.expand_dims(self.medoid, 0), centers, self.metric)[0]
        radii = [self.radius + c.radius for c in potential_neighbors]

        self.neighbors = {c: float(d) for c, d, r in zip(potential_neighbors, distances, radii) if d <= r}
        return self.neighbors

    def tree_search(self, point: Data, radius: Radius, depth: int) -> List['Cluster']:
        """ Searches down the tree for clusters that overlap point with radius at depth.
        """
        logging.debug(f'tree_search(point={point}, radius={radius}, depth={depth}')
        if depth == -1:
            depth = len(self.manifold.graphs)
        if depth < self.depth:
            raise ValueError('depth must not be less than cluster.depth')
        results = []
        if self.depth == depth:
            results = [self]
        elif self.overlaps(point, radius):
            results = self._tree_search(point, radius, depth)
        return results

    def _tree_search(self, point, radius, depth):
        assert self.overlaps(point, radius), f'_tree_search was started with no overlap.'
        assert self.depth < depth, f'_tree_search needs to have depth ({depth}) > self.depth ({self.depth}). '
        # results and candidates ONLY contain clusters that have overlap with point
        results: List[Cluster] = []
        candidates: List[Cluster] = [self]
        for d_ in range(self.depth, depth):
            results.extend([c for c in candidates if len(c.children) < 1])
            candidates = [c for c in candidates if len(c.children) > 0]
            children: List[Cluster] = [c for candidate in candidates for c in candidate.children]
            if len(children) == 0:
                break
            centers = np.asarray([c.medoid for c in children])
            distances = cdist(np.expand_dims(point, 0), centers, self.metric)[0]
            radii = [radius + c.radius for c in children]
            candidates = [c for c, d, r in zip(children, distances, radii) if d <= r]
            if len(candidates) == 0:
                break
        assert all((depth >= r.depth for r in results))
        assert all((depth == c.depth for c in candidates))
        return results

    def prune(self) -> None:
        """ Removes all references to descendents. """
        logging.debug(str(self))
        if self.children:
            [c.prune() for c in self.children]
            self.children = set()
        [c.neighbors.pop(c) for c in self.neighbors.keys()]
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
            # TODO: Can this be made more efficient? In the context of the larger manifold and graph
            logging.debug(f'{self} did not partition.')
            self.children = {
                Cluster(
                    self.manifold,
                    self.argpoints,
                    self.name + '0',
                    _argsamples=self.argsamples,
                    _argmedoid=self.argmedoid,
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
        [(p1_idx if p1 < p2 else p2_idx).append(i)
         for batch in iter(self)
         for i, p1, p2 in zip(batch, *cdist(poles, self.manifold.data[batch], self.metric))]

        # Ensure that p1 contains fewer points than p2
        p1_idx, p2_idx = (p1_idx, p2_idx) if len(p1_idx) < len(p2_idx) else (p2_idx, p1_idx)
        self.children = {
            Cluster(self.manifold, p1_idx, self.name + '1'),
            Cluster(self.manifold, p2_idx, self.name + '2'),
        }
        logging.debug(f'{self} was partitioned.')
        return self.children

    def distance(self, points: Data) -> List[Radius]:
        """ Returns the distance from self.medoid to every point in points. """
        return cdist(np.expand_dims(self.medoid, 0), points, self.metric)[0]

    def overlaps(self, point: Data, radius: Radius) -> bool:
        """ Checks if point is within radius + self.radius of cluster. """
        return self.distance(np.expand_dims(point, axis=0))[0] <= (self.radius + radius)

    def json(self):
        data = {
            'name': self.name,
            'argpoints': None,  # Do save them until at leaves.
            'children': [],
            '_radius': self.radius,
            '_argradius': self.argradius,
            '_argsamples': self.argsamples,
            '_argmedoid': self.argmedoid,
            '_local_fractal_dimension': self.local_fractal_dimension,
        }
        if self.children:
            data['children'] = [c.json() for c in self.children]
        else:
            data['argpoints'] = self.argpoints
        return data

    @staticmethod
    def from_json(manifold, data):
        children = set([Cluster.from_json(manifold, c) for c in data.pop('children', [])])
        return Cluster(manifold, children=children, **data)


class Graph:
    """ Graph comprised of clusters.

    Graphs hold sets of clusters, which are the vertices.
    The graph class is responsible for handling operations that occur
    solely within a layer of Manifold.graphs.
    """

    def __init__(self, *clusters):
        logging.debug(f'Graph(clusters={[str(c) for c in clusters]})')
        assert all(isinstance(c, Cluster) for c in clusters)
        assert all([c.depth == clusters[0].depth for c in clusters[1:]])
        self.clusters: Dict[Cluster: 'Graph'] = {c: None for c in clusters}
        return

    def __eq__(self, other: 'Graph') -> bool:
        return self.clusters.keys() == other.clusters.keys()

    def __iter__(self) -> Iterable[Cluster]:
        yield from self.clusters.keys()

    def __len__(self) -> int:
        # TODO: Consider __len__ -> cardinality
        return len(self.clusters.keys())

    def __str__(self) -> str:
        return ';'.join(sorted([str(c) for c in self.clusters.keys()]))

    def __repr__(self) -> str:
        return '\t'.join(sorted([repr(c) for c in self.clusters.keys()]))

    def __hash__(self):
        return hash(str(self))

    def __contains__(self, cluster: 'Cluster') -> bool:
        return cluster in self.clusters.keys()

    @property
    def manifold(self) -> 'Manifold':
        return next(iter(self.clusters.keys())).manifold if len(self.clusters.keys()) > 0 else None

    @property
    def depth(self) -> int:
        return next(iter(self.clusters.keys())).depth if len(self.clusters.keys()) > 0 else None

    def build_edges(self, threshold: int = None) -> None:
        """ Calculates and assigns neighbors for each cluster in the graph.

        A cluster can only be a neighbor with another cluster at the same depth.
        Two clusters are neighbors if and only if the two clusters have overlap.

        :param threshold: number of generations to look back to guarantee that all overlaps are found.
        """
        if threshold is None:  # assume manifold dimension (d) is less than the square root of embedding dimension (D).
            threshold = int(np.ceil(np.sqrt(self.manifold.data.shape[1])))  # Change this for when len(shape) > 2
        if threshold < 1:
            raise ValueError(f'threshold must be a positive integer. Got {threshold}')

        [cluster.update_neighbors(threshold) for cluster in self.clusters.keys()]
        return

    @property
    def edges(self) -> Dict[Set['Cluster'], float]:
        """ Returns all edges within the graph.
        """
        if '_edges' not in self.__dict__:
            logging.debug(f'building cache for {self}')
            self.__dict__['_edges'] = {frozenset([c, n]): d for c in self.clusters.keys() for n, d in c.neighbors.items()}
        return self.__dict__['_edges']

    @property
    def subgraphs(self) -> Set['Graph']:
        """ Returns all subgraphs within the graph.
        """
        if any((s is None for s in self.clusters.values())):
            unvisited = {c for c, s in self.clusters.items() if s is None}
            while unvisited:
                cluster = unvisited.pop()
                component = self.bft(cluster)
                unvisited -= component
                subgraph = Graph(*component)
                self.clusters.update({c: subgraph for c in subgraph})
        return set(self.clusters.values())

    def subgraph(self, cluster: 'Cluster') -> 'Graph':
        """ Returns the subgraph to which the cluster belongs. """
        if cluster not in self:
            raise ValueError(f'Cluster {cluster} not a member of {self}')

        if self.clusters[cluster] is None:
            component = self.bft(cluster)
            subgraph = Graph(*component)
            self.clusters.update({c: subgraph for c in subgraph})

        return self.clusters[cluster]

    def clear_cache(self) -> None:
        """ Clears the cache of the graph. """
        for prop in ['_edges']:
            logging.debug(str(self.clusters))
            try:
                del self.__dict__[prop]
            except KeyError:
                pass
        # Clear all cached subgraphs.
        self.clusters = {c: None for c in self.clusters.keys()}
        return

    @staticmethod
    def bft(start: 'Cluster'):
        """ Breadth-First Traversal starting at start. """
        logging.debug(f'starting from {start}')
        visited = set()
        queue = deque([start])
        while queue:
            c = queue.popleft()
            if c not in visited:
                visited.add(c)
                [queue.append(neighbor) for neighbor in c.neighbors.keys()]
        return visited

    @staticmethod
    def dft(start: 'Cluster'):
        """ Depth-First Traversal starting at start. """
        logging.debug(f'starting from {start}')
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

    def __init__(self, data: Data, metric: Metric, argpoints: Union[Vector, float] = None, **kwargs):
        logging.debug(f'Manifold(data={data.shape}, metric={metric}, argpoints={argpoints})')
        self.data: Data = data
        self.metric: Metric = metric

        if argpoints is None:
            self.argpoints = list(range(self.data.shape[0]))
        elif type(argpoints) is list:
            self.argpoints = list(map(int, argpoints))
        elif type(argpoints) is float:
            self.argpoints = np.random.choice(self.data.shape[0], int(self.data.shape[0] * argpoints), replace=False)
            self.argpoints = list(map(int, self.argpoints))
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

    def find_points(self, point: Data, radius: Radius) -> Dict[Data, Radius]:
        """ Returns all indices of points that are within radius of point. """
        candidates = [p for c in self.find_clusters(point, radius, len(self.graphs)) for p in c.argpoints]
        results: Dict[Data, Radius] = dict()
        point = np.expand_dims(point, axis=0)
        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i:i + BATCH_SIZE]
            distances = cdist(point, self.data[batch], self.metric)[0]
            results.update({p: d for p, d in zip(batch, distances) if d <= radius})
        return results

    def find_clusters(self, point: Data, radius: Radius, depth: int) -> Set['Cluster']:
        """ Returns all clusters that contain points within radius of point at depth. """
        return {r for c in self.graphs[0] for r in c.tree_search(point, radius, depth)}

    def build(self, *criterion) -> 'Manifold':
        """ Rebuilds the stack of graphs. """
        self.graphs = [Graph(Cluster(self, self.argpoints, ''))]
        self.build_tree(*criterion)
        self.build_graphs()
        return self

    def build_tree(self, *criterion) -> 'Manifold':
        """ Builds the cluster tree. """
        while True:
            logging.info(f'current depth: {len(self.graphs) - 1}')
            clusters = self._partition_threaded(criterion)
            if len(self.graphs[-1]) < len(clusters):
                g = Graph(*clusters)
                self.graphs.append(g)
            else:
                [c.children.clear() for c in self.graphs[-1]]
                break
        return self

    def build_graphs(self) -> 'Manifold':
        """ Builds the graphs. """
        [g.build_edges() for g in self.graphs]
        return self

    def subgraph(self, cluster: Union[str, Cluster]) -> Graph:
        """ Returns the subgraph to which cluster belongs. """
        cluster = self.select(cluster) if type(cluster) is str else cluster
        return self.graphs[cluster.depth].subgraph(cluster)

    def graph(self, cluster: Union[str, Cluster]) -> Graph:
        """ Returns the graph to which cluster belongs. """
        cluster = self.select(cluster) if type(cluster) is str else cluster
        return self.graphs[cluster.depth]

    def _partition_single(self, criterion):
        return [child for cluster in self.graphs[-1] for child in cluster.partition(*criterion)]

    def _partition_threaded(self, criterion):
        queue = Queue()
        threads = [
            Thread(
                target=lambda cluster: [queue.put(c) for c in cluster.partition(*criterion)],
                args=(c,),
                name=c.name
            )
            for c in self.graphs[-1]]
        [t.start() for t in threads]
        [t.join() for t in threads]
        clusters = []
        while not queue.empty():
            clusters.append(queue.get())
        return clusters

    def select(self, name: str) -> Cluster:
        """ Returns the cluster with the given name. """
        assert len(name) <= len(self.graphs)
        cluster: Cluster = next(iter(self.graphs[0]))
        for depth in range(len(name) + 1):
            partial_name = name[:depth]
            for child in cluster.children:
                if child.name == partial_name:
                    cluster = child
                    break
        assert name == cluster.name, f'wanted {name} but got {cluster.name}.'
        return cluster

    def dump(self, fp: TextIO) -> None:
        pickle.dump({
            'metric': self.metric,
            'argpoints': self.argpoints,
            'root': [c.json() for c in self.graphs[0]],
        }, fp)
        return

    @staticmethod
    def load(fp: TextIO, data: Data) -> 'Manifold':
        d = pickle.load(fp)
        manifold = Manifold(data, metric=d['metric'], argpoints=d['argpoints'])
        graphs = [
            Graph(*[Cluster.from_json(manifold, r) for r in d['root']])
        ]
        while True:
            layer = Graph(*(child for cluster in graphs[-1] for child in cluster.children))
            if not layer:
                break
            else:
                graphs.append(layer)

        manifold.graphs = graphs
        return manifold
