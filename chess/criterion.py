import chess
from chess.manifold import Cluster as _Cluster, Manifold as _Manifold


class MaxDepth:
    """ Allows clustering up until the given depth.
    """

    def __init__(self, depth):
        self.depth = depth

    def __call__(self, cluster: _Cluster):
        return cluster.depth < self.depth


class AddLevels:  # TODO: Cover
    """ Allows clustering up until current.depth + depth.
    """

    def __init__(self, depth):
        self.depth = depth
        self.start = None

    def __call__(self, cluster: _Cluster):
        if self.start is None:
            self.start = cluster.depth
        return cluster.depth < (self.start + self.depth)


class MinPoints:
    """ Allows clustering up until there are fewer than points.
    """

    def __init__(self, points):
        self.points = points

    def __call__(self, cluster: _Cluster):
        return len(cluster) > self.points


class MinRadius:
    """ Allows clustering until cluster.radius is less than radius.
    """

    def __init__(self, radius):
        self.radius = radius
        chess.manifold.MIN_RADIUS = radius

    def __call__(self, cluster: _Cluster):
        if cluster.radius <= self.radius:
            cluster.__dict__['_min_radius'] = self.radius
            return False
        return True


class LeavesComponent:
    """ Allows clustering until the cluster has left the component of the parent.
    """

    def __init__(self, manifold: _Manifold):
        self.manifold = manifold
        return

    def __call__(self, cluster: _Cluster):
        parent_component = self.manifold.graphs[cluster.depth - 1].component(self.manifold.select(cluster.name[:-1]))
        return any((c.overlaps(cluster.medoid, cluster.radius) for c in parent_component))


class MinCardinality:
    """ Allows clustering until cardinality of cluster's component is less than given.
    """

    def __init__(self, cardinality):
        self.cardinality = cardinality

    def __call__(self, cluster: _Cluster):
        return len(cluster.manifold.graphs[cluster.depth].component(cluster)) > self.cardinality


class MinNeighborhood:
    """ Allows clustering until the size of the neighborhood drops below threshold.
    """

    def __init__(self, starting_depth: int, threshold: int):
        self.starting_depth = starting_depth
        self.threshold = threshold
        return

    def __call__(self, cluster: _Cluster) -> bool:
        return cluster.depth < self.starting_depth or len(cluster.neighbors) >= self.threshold


class NewComponent:
    """ Cluster until a new component is created. """

    def __init__(self, manifold: _Manifold):
        self.manifold = manifold
        self.starting = len(manifold.graphs[-1].components)
        return

    def __call__(self, _):
        return len(self.manifold.graphs[-1].components) == self.starting
