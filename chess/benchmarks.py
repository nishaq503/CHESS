import logging
from inspect import isclass

import click as click
import mlflow

from chess.query import Query
from chess.search import linear_search
from . import CHESS, datasets

log = logging.getLogger(__name__)


def _load(dataset: str) -> datasets.Dataset:
    try:
        dataset = eval(f'datasets.{dataset}')
    except AttributeError as e:
        options = list(filter(
            lambda s: isclass(s) and issubclass(s, datasets.Dataset) and s is not datasets.Dataset,
            [getattr(datasets, s) for s in dir(datasets)]
        ))
        log.exception(f'Invalid dataset selected. Choices are {options}')
        raise
    return dataset


@click.command()
@click.argument('dataset')
@click.argument('metric')
@click.argument('radius')
def search(dataset, metric, radius):
    data = _load(dataset).get_data()

    # Clustering Runtime.
    with mlflow.start_run(experiment_id="Clustering"):
        chess = CHESS(data, metric)
    queries = dataset.get_queries()

    # Searching.
    with mlflow.start_run(experiment_id="Clustered Search"):
        for query in queries:
            chess.search(query, radius)

    with mlflow.start_run(experiment_id="Linear Search"):
        for query in queries:
            linear_search(chess.cluster, Query(point=query, radius=radius))
