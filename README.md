# Clustered Hierarchical Entropy-Scaling Search of Astrophysical and Biological Data

CHESS is a search algorithm for large data sets when the data exhibits certain geometric properties.
The [paper](https://arxiv.org/abs/1908.08551.pdf) is available on the arXiv.

We have extended CHESS to perform Manifold Learning and Anomaly Detection.
We are working on adding Dimensionality Reduction and Visualization abilities, and on 3-d Object Recognition from point clouds.
One of the major problems with this extension is that we need a new name.
Stay tuned.

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/nishaq503/CHESS.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/nishaq503/CHESS/context:python)
[![codecov](https://codecov.io/gh/thoward27/chess/branch/master/graph/badge.svg)](https://codecov.io/gh/thoward27/chess)
[![Documentation Status](https://readthedocs.org/projects/clustered-hierarchical-entropy-scaling-search/badge/?version=latest)](https://clustered-hierarchical-entropy-scaling-search.readthedocs.io/en/latest/?badge=latest)

## Installation

```bash
python3 -m pip install CHESS-python
```

## Usage

```python
import numpy as np

from chess.datasets import bullseye
from chess.manifold import Manifold
from chess import criterion

# Get the data.
data, _ = bullseye()
# data is a numpy.ndarray in this case but it could just as easily be a numpy.memmap if your data cannot fit in RAM.
# We used memmaps for the research, though it does impose file-io costs.

manifold = Manifold(data=data, metric='euclidean')
# Any metric allowed by scipy's cdist function is allowed in Manifold.
# You can also define your own distance function. It will work so long as scipy allows it.

manifold.build(criterion.MaxDepth(20), criterion.MinRadius(0.25))
# Manifold.build can optionally take any number of early stopping criteria.
# chess.criterion defines some criteria that we have used in research.
# You are free to define your own.
# Take a look at chess/criterion.py for hints of how to define custom criteria.

# A sample rho-nearest neighbors search query
query, radius = data[0], 0.05
results = manifold.find_points(point=query, radius=radius)
# results is a dictionary of indexes of hits in data and the distance to those hits.

# A sample k-nearest neighbors search query
results = manifold.find_knn(point=query, k=25)
```

chess.Manifold relies on the Graph and Cluster classes.
You can import these and work with them directly if you so choose.
We have written good docs for each class and method.
Go crazy.

## Contributing

Pull requests and bug reports are welcome.
For major changes, please first open an issue to discuss what you would like to change.

## License

[MIT](LICENSE)
