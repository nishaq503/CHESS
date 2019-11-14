# Clustered Hierarchical Entropy-Scaling Search of Astrophysical and Biological Data

CHESS is a search algorithm for large data sets when the data exhibits certain geometric properties.

All classes and functions are very well documented in the code. If you want more details, read them.
The [paper](https://arxiv.org/pdf/1908.08551.pdf) is available on the arXiv.


## Installation

Download this repository...

```bash
git clone https://github.com/nishaq503/CHESS.git
```

## Usage

```python
from src.search import Search

search_object = Search(dataset=..., metric=...)
search_object.build(depth=...)
results = search_object.search(query=..., radius=...)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)