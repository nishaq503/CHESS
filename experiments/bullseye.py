from chess import criterion
from chess.datasets import *
from chess.manifold import *

data = bullseye()
manifold = Manifold(data[0], 'euclidean')
manifold.build(criterion.NewComponent(manifold))
manifold.deepen(criterion.NewComponent(manifold))
assert len(manifold.graphs[-1].components) == 4, len(manifold.graphs[-1].components)
