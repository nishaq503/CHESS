""" Clustered Hierarchical Entropy-Scaling Search.
"""
from functools import partial
from multiprocessing.pool import Pool
from typing import List

from chess.cluster import Cluster
from chess.distance import calculate_distances
from chess.query import Query


def search(cluster: Cluster, query: Query) -> List[int]:
    """ Finds all points within query.radius of query.point.

    :param cluster: cluster to start searching in.
    :param query: point to search around.

    TODO: Benchmark search using all methods below and a method involving creating a new cluster from all points
    """
    clusters = cluster_search(cluster, query)

    # Multiprocessing.
    with Pool() as m:
        results = m.map(partial(linear_search, query=query), clusters)
    points = [p for result in results for p in result]

    # Using map
    # points = [p for results in map(partial(linear_search, query=query), clusters) for p in results]

    # Comprehension
    # points = [p for cluster in clusters for p in linear_search(cluster, query)]
    return points


def cluster_search(cluster: Cluster, query: Query) -> List[Cluster]:
    """ Find all clusters that may contain hits.

    :param cluster: the cluster to start searching in.
    :param query: point to search around.

    :return: List of names of clusters that may contain hits.
    """
    results = []
    if query in cluster:
        if cluster.radius() <= query.radius or not (cluster.left or cluster.right):
            # If our search radius is larger than the cluster,
            # or if the cluster has no children.
            results.append(cluster)

        elif cluster.depth < query.max_depth and (cluster.left or cluster.right):
            # Here, we can keep filtering down to children.
            if query in cluster.left:
                results.extend(cluster_search(cluster.left, query))
            if query in cluster.right:
                results.extend(cluster_search(cluster.right, query))

    return results


def linear_search(cluster: Cluster, query: Query) -> List[int]:
    """ Perform naive linear search on the clusters data.

    :param cluster: cluster to search within
    :param query: point around which to search.

    :return: list of indexes in cluster.data of hits.
    """
    results = []
    for i, batch in enumerate(cluster):
        distances = calculate_distances([query.point], batch, cluster.metric)[0]
        results.extend([i + j for j, d in enumerate(distances) if d <= query.radius])

    return results
