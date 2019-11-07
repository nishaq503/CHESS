from typing import Dict, List, Tuple

import numpy as np

from src import globals
from src.cluster import Cluster
from src.distance_functions import check_input_array, calculate_distances


def get_data_and_queries(
        dataset: str,
        mode: str = 'r',
) -> Tuple[np.memmap, np.memmap]:
    """
    Reads the numpy memmap files for the given data set and returns them.
    :param dataset: data set to read. Must be APOGEE or GreenGenes.
    :param mode: optional mode to read the memmap files in.
    :return: data for clustering, queries that were held out.
    """
    data: np.memmap
    queries: np.memmap

    if dataset == 'APOGEE':
        data = np.memmap(
            filename=globals.APOGEE_DATA,
            dtype=globals.APOGEE_DTYPE,
            mode=mode,
            shape=globals.APOGEE_DATA_SHAPE,
        )
        queries = np.memmap(
            filename=globals.APOGEE_QUERIES,
            dtype=globals.APOGEE_DTYPE,
            mode=mode,
            shape=globals.APOGEE_QUERIES_SHAPE,
        )
    elif dataset == 'GreenGenes':
        data = np.memmap(
            filename=globals.GREENGENES_DATA,
            dtype=globals.GREENGENES_DTYPE,
            mode=mode,
            shape=globals.GREENGENES_DATA_SHAPE,
        )
        queries = np.memmap(
            filename=globals.GREENGENES_QUERIES,
            dtype=globals.GREENGENES_DTYPE,
            mode=mode,
            shape=globals.GREENGENES_QUERIES_SHAPE,
        )
    else:
        raise ValueError(f'Only the APOGEE and GreenGenes datasets are available. Got {dataset}.')

    return data, queries


class Search:
    """
    Implements Clustered Hierarchical Entropy-Scaling Search with GPU acceleration using tensorflow.
    I'm trying to squash a few bugs right now so tensorflow implementation is currently turned off.
    """

    def __init__(
            self,
            dataset: str,
            metric: str,
            names_file: str = None,
            info_file: str = None,
            reading: bool = False,
    ):
        """
        Initializes search object.

        :param dataset: name of data to search.
        :param metric: distance metric to use during clustering and search.
        :param names_file: name of .csv with columns {cluster_name, point_index}.
        :param info_file: name of .csv with columns {cluster_name, number_of_points, center, radius, lfd, is_leaf}.
        :param reading: weather or not the cluster-tree for the search object is being read from a file.
        """
        self.dataset = dataset
        data, _ = get_data_and_queries(dataset=self.dataset)
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

        self.cluster_dict: Dict[str: Cluster] = self._get_cluster_dict()

    def _get_cluster_dict(self):
        cluster_dict: Dict[str: Cluster] = {}

        def in_order(node: Cluster):
            cluster_dict[node.name] = node
            if node.left:
                in_order(node.left)
            if node.right:
                in_order(node.right)

        in_order(self.root)
        return cluster_dict

    def _get_batch(
            self,
            points: List[int],
            start_index: int,
    ):
        num_points = min(start_index + globals.BATCH_SIZE, self.data.shape[0]) - start_index
        return np.asarray([self.data[p] for p in points[start_index: start_index + num_points]])

    def linear_search(
            self,
            query: np.ndarray,
            radius: float,
    ) -> List[int]:
        """
        Perform naive linear search on self.data.
        This is to get ground truth for clustered search to compare against.

        :param query: point around which to search.
        :param radius: search radius to use.
        :return: list of indexes in self.data of hits.
        """
        check_input_array(query)

        results = []
        points = list(range(self.data.shape[0]))

        for i in range(0, self.data.shape[0], globals.BATCH_SIZE):
            batch = self._get_batch(points, i)
            distances = calculate_distances(query, batch, self.metric)[0, :]
            results.extend([i + j for j, d in enumerate(distances) if d <= radius])

        return results

    def search(
            self,
            query: np.ndarray,
            radius: float,
            search_depth: int,
            count_calls: bool = False,
    ) -> Tuple[List[int], int, float]:
        """
        Perform clustered search to required depth.

        :param query: point around which to search.
        :param radius: search radius to use.
        :param search_depth: maximum depth to which to search.
        :param count_calls: weather or not to count distance calls for benchmarking.
        :return: List of indexes in self.data of hits, number of clusters searched, fraction of dataset searched.
        """
        check_input_array(query)

        clusters = self.root.search(query, radius, search_depth)
        potential_hits = [p
                          for c in clusters
                          for p in self.cluster_dict[c].points]

        results = []

        for i in range(0, len(potential_hits), globals.BATCH_SIZE):
            batch = self._get_batch(potential_hits, i)
            distances = calculate_distances(query, batch, self.metric, count_calls=count_calls)[0, :]
            hits = [i + j for j, d in enumerate(distances) if d <= radius]
            results.extend([potential_hits[h] for h in hits])

        return results, len(clusters), len(potential_hits) / self.data.shape[0]

    def print_names(
            self,
            filename: str = None
    ):
        """
        Write a .csv with columns {cluster_name, point_index}.
        :param filename: Optional file to write to.
        """
        filename = self.names_file if filename is None else filename
        with open(filename, 'w') as outfile:
            outfile.write('cluster_name,point\n')

            def _helper(node: Cluster):
                if node.left:
                    _helper(node.left)
                if node.right:
                    _helper(node.right)
                else:
                    [outfile.write(f'{node.name},{str(p)}\n') for p in node.points]
                    outfile.flush()

            _helper(self.root)
        return

    def print_info(
            self,
            filename: str = None
    ):
        """
        Write a .csv with columns {cluster_name, number_of_points, center, radius, lfd, is_leaf}.
        :param filename: Optional file to write to.
        """
        filename = self.info_file if filename is None else filename
        with open(filename, 'w') as outfile:
            outfile.write('cluster_name,number_of_points,center,radius,lfd,is_leaf\n')

            def _helper(node: Cluster):
                outfile.write(f'{node.name},{len(node.points):d},{node.center:d},{node.radius:.16f},'
                              f'{node.local_fractal_dimension:.16f},{not node.can_be_popped()}\n')
                if node.left:
                    _helper(node.left)
                if node.right:
                    _helper(node.right)

            _helper(self.root)
        return

    def read_cluster_tree(
            self,
            names_file: str = None,
            info_file: str = None,
    ) -> Cluster:
        """
        Reads a cluster-tree from (optionally given) .csv files on disk

        :param names_file: name of .csv with columns {cluster_name, point_index}.
        :param info_file: name of .csv with columns {cluster_name, number_of_points, center, radius, lfd, is_leaf}.
        :return: root cluster.
        """
        self.names_file = names_file if names_file is not None else self.names_file
        name_to_points: Dict[str: List[int]] = {}

        with open(self.names_file, 'r') as infile:
            infile.readline()
            while True:  # this is an obvious use-case for walrus operator in python3.8
                line = infile.readline()
                if not line:
                    break

                line = line.split(',')
                cluster_name, point = line[0], int(line[1])
                if cluster_name in name_to_points.keys():
                    name_to_points[cluster_name].append(point)
                else:
                    name_to_points[cluster_name] = [point]

        def _build_dict_tree(name: str) -> List[int]:
            if name in name_to_points.keys():
                return name_to_points[name]
            else:
                left = _build_dict_tree(f'{name}0')
                right = _build_dict_tree(f'{name}1')
                name_to_points[name] = left.copy() + right.copy()
                return name_to_points[name]

        _build_dict_tree('')

        self.info_file = info_file if info_file is not None else self.info_file
        name_to_info: Dict[str, List] = {}
        with open(self.info_file, 'r') as infile:
            infile.readline()
            while True:
                line = infile.readline()
                if not line:
                    break

                [cluster_name, num_points, center, radius, lfd, is_leaf] = line.split(',')
                if cluster_name not in name_to_points.keys():
                    raise ValueError(f'{cluster_name} not found in name_to_points dictionary.')

                name_to_info[cluster_name] = [int(center), float(radius), float(lfd), bool(is_leaf)]
                if len(name_to_points[cluster_name]) != int(num_points):
                    raise ValueError(f'Mismatch in number of points in cluster {cluster_name}.\n'
                                     f'Got {num_points} from {self.info_file}.\n'
                                     f'Got {len(name_to_points[cluster_name])} from {self.names_file}.')

        def _build_tree(name: str) -> Cluster:
            return Cluster(
                data=self.data,
                points=name_to_points[name].copy(),
                metric=self.metric,
                name=name,
                center=name_to_info[name][0],
                radius=name_to_info[name][1],
                local_fractal_dimension=name_to_info[name][0],
                left=_build_tree(f'{name}0'),
                right=_build_tree(f'{name}1'),
                reading=True,
            ) if name in name_to_points else None

        return _build_tree('')

    def cluster_deeper(
            self,
            new_depth: int,
    ):
        old_depth = max(list(map(len, self.cluster_dict.keys())))

        [cluster.pop(update=True)
         for i in range(old_depth, new_depth)
         for name, cluster in self.cluster_dict.items()
         if len(cluster.name) == i]

        self.cluster_dict = self._get_cluster_dict()
        return
