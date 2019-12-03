from chess.cluster import Cluster


class MaxDepth:
    def __init__(self, value):
        self.value = value

    def __call__(self, cluster: Cluster):
        return cluster.depth < self.value


class AddLevels:
    def __init__(self, value):
        self.value = value
        self.first = None

    def __call__(self, cluster: Cluster):
        if self.first is None:
            self.first = cluster.depth
        return cluster.depth < (self.first + self.value)


class MinPoints:
    def __init__(self, value):
        self.value = value

    def __call__(self, cluster: Cluster):
        return cluster.n > self.value


class MinRadius:
    def __init__(self, value):
        self.value = value

    def __call__(self, cluster: Cluster):
        return cluster.radius > self.value


class ManifoldShattering:
    def __init__(self, value):
        self.value = value

    def __call__(self, cluster: Cluster):
        return not cluster.manifold.shattered
