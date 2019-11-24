import logging

import click as click
import mlflow

from benchmarks.datasets import load
from chess import CHESS
from chess.query import Query
from chess.search import linear_search

log = logging.getLogger(__name__)


@click.command()
@click.argument('dataset')
@click.argument('metric')
@click.argument('radius')
def search(dataset, metric, radius):
    data = load(dataset).get_data()

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
            linear_search(chess.root, Query(point=query, radius=radius))
