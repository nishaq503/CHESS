from typing import Dict

import numpy as np

from src import globals
from src.cluster import Cluster


class Search:
    """
    Implements Clustered Hierarchical Entropy-Scaling Search with GPU acceleration using tensorflow.
    I'm trying to squash a few bugs right now so tensorflow implementation is currently turned off.
    """

    def __init__(
            self,
            data: np.memmap,
            metric: str,
            names_file: str = None,
            info_file: str = None,
            reading: bool = False,
    ):
        """
        Initializes search object.

        :param data: numpy.memmap of data to search.
        :param metric: distance metric to use during clustering and search.
        :param names_file: name of .csv with columns {cluster_name, point_index}.
        :param info_file: name of .csv with columns {cluster_name, number_of_points, center, radius, lfd, is_leaf}.
        :param reading: weather or not the cluster-tree for the search object is being read from a file.
        """

        self.data: np.memmap = data

        if metric not in globals.DISTANCE_FUNCTIONS:
            raise NotImplementedError(f'Got metric {metric}. It must be one of {globals.DISTANCE_FUNCTIONS}.')
        self.metric: str = metric

        self.names_file: str = names_file
        self.info_file: str = info_file

        self.root: Cluster

        if reading:
            self.root = self.read_cluster_tree()
        else:
            self.root = Cluster(
                data=self.data,
                points=list(range(self.data.shape[0])),
                metric=self.metric,
                name='',
            )
            self.root.make_tree()

        self._cluster_dict: Dict[str: Cluster] = self._get_cluster_dict()

    def _get_cluster_dict(self) -> Dict[str: Cluster]:
        cluster_dict: Dict[str: Cluster] = {}

        def in_order(node: Cluster):
            cluster_dict[node.name] = node
            if node.left:
                in_order(node.left)
            if node.right:
                in_order(node.right)

        in_order(self.root)
        return cluster_dict
