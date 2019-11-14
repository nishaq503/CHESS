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
    Implements Clustered Hierarchical Entropy-Scaling Search.
    All it needs is a dataset and a distance function (preferably a metric).
    """

    def __init__(
            self,
            dataset: str,
            metric: str,
    ):
        """
        Initializes search object.

        :param dataset: name of dataset to search.
        :param metric: distance metric to use during clustering and search.
        """
        self.dataset = dataset
        data, _ = get_data_and_queries(dataset=self.dataset)
        self.data: np.memmap = data

        if metric not in globals.DISTANCE_FUNCTIONS:
            raise NotImplementedError(f'Got metric {metric}. It must be one of {globals.DISTANCE_FUNCTIONS}.')
        self.metric: str = metric

        self.root: Cluster
        self.cluster_dict: Dict[str: Cluster]
        self.names_file: str
        self.info_file: str

    # noinspection PyAttributeOutsideInit
    def build(self, depth: int):
        """ Builds cluster-tree. """
        self.root = Cluster(
            data=self.data,
            points=list(range(self.data.shape[0])),
            metric=self.metric,
            name='',
        )
        self.partition(depth=depth)
        self.cluster_dict = self._get_cluster_dict()
        return

    def partition(
            self,
            depth: int,
    ):
        old_depth = max(map(len, self.cluster_dict.keys()))

        [cluster.pop(update=True)
         for i in range(old_depth, depth)
         for name, cluster in self.cluster_dict.items()
         if len(name) == i]

        # noinspection PyAttributeOutsideInit
        self.cluster_dict = self._get_cluster_dict()
        return

    # noinspection PyAttributeOutsideInit
    def load(
            self,
            names_file: str,
            info_file: str,
    ):
        """ Loads cluster-tree from files on disk. """
        self.names_file = names_file
        self.info_file = info_file
        self.root = self.read_cluster_tree()
        self.cluster_dict = self._get_cluster_dict()
        return

    def write(
            self,
            names_file: str,
            info_file: str,
    ):
        self._print_names(filename=names_file)
        self._print_info(filename=info_file)
        return

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
            batch_size: int = globals.BATCH_SIZE,
    ):
        """
        Gets a batch of points from the given points list.
        Batch starts at index start_index in points list.

        :param points: list of indexes in self.data from which to draw the batch.
        :param start_index: index in points from where to start drawing the batch.
        :param batch_size: size of each batch.

        :return numpy array of points in the batch:
        """
        num_points = min(start_index + batch_size, self.data.shape[0]) - start_index
        return np.asarray([self.data[p] for p in points[start_index: start_index + num_points]])

    def linear_search(
            self,
            query: np.ndarray,
            radius: globals.RADII_DTYPE,
    ) -> List[int]:
        """
        Perform naive linear search on self.data.
        This is for comparing against clustered search.

        :param query: point around which to search.
        :param radius: search radius to use.
        :return: list of indexes in self.data of hits.
        """
        check_input_array(query)

        results = []
        points = list(range(self.data.shape[0]))

        for i in range(0, self.data.shape[0], globals.BATCH_SIZE):
            distances = calculate_distances(query, self._get_batch(points, i), self.metric)[0, :]
            results.extend([i + j for j, d in enumerate(distances) if d <= radius])

        return results

    def search(
            self,
            query: np.ndarray,
            radius: globals.RADII_DTYPE,
            search_depth: int = globals.MAX_DEPTH,
            count: bool = False,
    ):
        """
        Perform clustered search to required depth.

        :param query: point around which to search. This must have shape (1, num_dims)
        :param radius: search radius to use.
        :param search_depth: maximum depth to which to search.
        :param count: weather or not to count distance calls for benchmarking.
        :return: List of indexes in self.data of hits.
        """
        check_input_array(query)

        clusters = self.root.search(query, radius, search_depth)
        potential_hits = [p for c in clusters for p in self.cluster_dict[c].points]

        results = []

        for i in range(0, len(potential_hits), globals.BATCH_SIZE):
            distances = calculate_distances(
                x=query,
                y=self._get_batch(potential_hits, i),
                metric=self.metric,
                count_calls=count,
            )[0, :]
            results.extend([potential_hits[i + j] for j, d in enumerate(distances) if d <= radius])

        if count:
            return results, len(clusters), len(potential_hits) / self.data.shape[0]
        else:
            return results

    def _print_names(
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

    def _print_info(
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
                outfile.write(f'{node.name},{len(node.points):d},{node.center:d},{node.radius:.8f},'
                              f'{node.local_fractal_dimension:.8f},{not node.can_be_popped()}\n')
                if node.left:
                    _helper(node.left)
                if node.right:
                    _helper(node.right)

            _helper(self.root)
        return

    def read_cluster_tree(self) -> Cluster:
        """
        Reads a cluster-tree from (optionally given) .csv files on disk

        :return: root cluster.
        """
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

                name_to_info[cluster_name] = [int(center), globals.RADII_DTYPE(radius),
                                              globals.RADII_DTYPE(lfd), bool(is_leaf)]
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
                local_fractal_dimension=name_to_info[name][2],
                left=_build_tree(f'{name}0'),
                right=_build_tree(f'{name}1'),
                reading=True,
            ) if name in name_to_points else None

        return _build_tree('')
