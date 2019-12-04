from chess import criterion
from chess.datasets import *
from chess.manifold import *

data = bullseye()
manifold = Manifold(data[0], 'euclidean')
manifold.build(criterion.MinRadius(1.))
assert len(manifold.graphs[-1].components) == 4, len(manifold.graphs[-1].components)
