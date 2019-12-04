from chess.manifold import Cluster, Manifold


class MaxDepth:
    """ Allows clustering up until the given depth.
    """

    def __init__(self, depth):
        self.depth = depth

    def __call__(self, cluster: Cluster):
        return cluster.depth < self.depth


class AddLevels:
    """ Allows clustering up until current.depth + depth.
    """

    def __init__(self, depth):
        self.depth = depth
        self.start = None

    def __call__(self, cluster: Cluster):
        if self.start is None:
            self.start = cluster.depth
        return cluster.depth < (self.start + self.depth)


class MinPoints:
    """ Allows clustering up until there are fewer than points.
    """

    def __init__(self, points):
        self.points = points

    def __call__(self, cluster: Cluster):
        return len(cluster) > self.points


class MinRadius:
    """ Allows clustering until cluster.radius is less than radius.
    """

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, cluster: Cluster):
        return cluster.radius > self.radius


class LeavesComponent:
    """ Allows clustering until the cluster has left the component of the parent.
    """

    def __init__(self, manifold: Manifold):
        self.manifold = manifold
        return

    def __call__(self, cluster: Cluster):
        parent_component = self.manifold.graphs[cluster.depth].component(cluster)
        return any((c.overlaps(cluster.center, cluster.radius) for c in parent_component))


class MinCardinality:
    """ Allows clustering until cardinality of cluster's component is less than given.
    """

    def __init__(self, cardinality):
        self.cardinality = cardinality

    def __call__(self, cluster: Cluster):
        return len(cluster.manifold.graphs[cluster.depth].component(cluster)) > self.cardinality
