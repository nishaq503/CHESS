import os
import pickle
import zipfile
from typing import List, Tuple, Dict

import numpy as np

import old_chess.config as config
from old_chess.src.cluster import Cluster
from old_chess.src.utils import tf_calculate_distance


class Search:
    """ Implements entropy-scaling search with hierarchical clustering with GPU acceleration. """

    def __init__(
            self,
            data: np.memmap,
            distance_function: str,
            reading: bool = False,
            names_file: str = None,
            info_file: str = None,
    ):
        """ Initializes search object.

        :param data: numpy.memmap of data to search.
        :param distance_function: distance function to use.
        :param reading: weather the search object is to be read from file.
        :param names_file: filename of the csv containing: cluster_name, point_index.
        :param info_file: filename of the csv containing: name, number_of_points, center, radius, lfd, is_leaf.
        """
        self.data: np.memmap = data
        self.distance_function: str = distance_function

        if reading:
            self.names_file: str = names_file
            self.info_file: str = info_file
            self.root = self.read_cluster_tree()
        else:
            self.root: Cluster = Cluster(
                data=self.data,
                points=list(range(np.shape(self.data)[0])),
                distance_function=self.distance_function,
                name='',
            )
            self.root.make_tree()

        self.cluster_dict_: Dict[str: Cluster] = self._get_cluster_dict()
        return

    def _get_cluster_dict(self):
        cluster_dict = {}

        def in_order(node: Cluster):
            cluster_dict[node.name] = node
            if node.left or node.right:
                in_order(node.left)
                in_order(node.right)

        in_order(self.root)
        return cluster_dict

    def linear_search(self, query: np.ndarray, radius: float) -> List[int]:
        """ Perform linear search for comparison with clustered search.

        :param query: point around with to search.
        :param radius: search radius to use.
        :return: list of indexes of hits.
        """
        results = []

        for i in range(0, np.shape(self.data)[0], config.BATCH_SIZE):
            end = min(i + config.BATCH_SIZE, np.shape(self.data)[0])
            batch = np.asarray([self.data[j] for j in range(i, end)])
            distances = tf_calculate_distance(query, batch, self.distance_function)
            results.extend([i + j for j, d in enumerate(distances) if d <= radius])

        return results

    def clustered_search(self, query: np.ndarray, radius: float, search_depth: int) -> Tuple[List[int], int, float]:
        """ Perform clustered search.

        :param query: point around with to search.
        :param radius: search radius to use.
        :param search_depth: maximum depth to which to search.
        :return: list of indexes of hits.
        """
        clusters = self.root.search(query, radius, search_depth)
        clustered_space = [self.cluster_dict_[c].points for c in clusters]
        search_space = [p for points in clustered_space for p in points]
        fraction_searched = len(search_space) / np.shape(self.data)[0]

        results = []
        for i in range(0, len(search_space), config.BATCH_SIZE):
            end = min(i + config.BATCH_SIZE, len(search_space))
            batch = np.asarray([self.data[search_space[p]] for p in range(i, end)])
            distances = tf_calculate_distance(query, batch, self.distance_function)
            hits = [i + j for j, d in enumerate(distances) if d <= radius]
            results.extend([search_space[h] for h in hits])
        return results, len(clusters), fraction_searched

    def print_names(self, filename: str) -> None:
        """ Print .csv file containing: cluster_name, point_index. """

        with open(filename, 'w') as outfile:
            outfile.write('name,point\n')

            def _names_helper(cluster: Cluster):
                if cluster.left or cluster.right:
                    _names_helper(cluster.left)
                    _names_helper(cluster.right)
                else:
                    [outfile.write(f'{cluster.name},{str(p)}\n') for p in cluster.points]
                    outfile.flush()

            _names_helper(self.root)
        return

    def print_info(self, filename: str) -> None:
        """ Print .csv file containing: name, number_of_points, center, radius, lfd, is_leaf. """

        with open(filename, 'w') as outfile:
            outfile.write('name,number_of_points,center,radius,lfd,is_leaf\n')

            def _info_helper(cluster: Cluster):
                outfile.write(f'{cluster.name},{len(cluster.points):d},{cluster.center:d},'
                              f'{cluster.radius:.8f},{cluster.lfd:.8f},{not cluster.can_be_popped()}\n')
                outfile.flush()
                if cluster.left or cluster.right:
                    _info_helper(cluster.left)
                    _info_helper(cluster.right)
                return

            _info_helper(self.root)
        return

    def read_cluster_tree(self) -> Cluster:
        """ Read cluster_tree from csv files on disk. """

        name_to_points = {}
        with open(self.names_file, 'r') as infile:
            infile.readline()
            while True:
                line = infile.readline()
                if not line:
                    break

                line = line.split(',')
                if line[0] in name_to_points:
                    name_to_points[line[0]].append(int(line[1]))
                else:
                    name_to_points[line[0]] = [int(line[1])]

        def build_dict_tree(name: str) -> List[int]:
            if name in name_to_points:
                return name_to_points[name]
            else:
                left = build_dict_tree(f'{name}1')
                right = build_dict_tree(f'{name}2')
                name_to_points[name] = left.copy() + right.copy()
                return name_to_points[name]

        build_dict_tree('')

        name_to_info = {}
        with open(self.info_file, 'r') as infile:
            infile.readline()
            while True:
                line = infile.readline()
                if not line:
                    break

                line = line.split(',')
                [center, radius, lfd, is_leaf] = line[2:]
                name_to_info[line[0]] = [int(center), np.float64(radius), float(lfd), bool(is_leaf)]

        def build_cluster_tree(name: str) -> Cluster:
            if name in name_to_points:
                left = build_cluster_tree(f'{name}1')
                right = build_cluster_tree(f'{name}2')

                return Cluster(
                    data=self.data,
                    points=name_to_points[name].copy(),
                    distance_function=self.distance_function,
                    name=name,
                    center=name_to_info[name][0],
                    radius=name_to_info[name][1],
                    lfd=name_to_info[name][2],
                    left=left,
                    right=right,
                    reading=True,
                )
            else:
                # noinspection PyTypeChecker
                return None

        return build_cluster_tree('')

    def cluster_deeper(self, new_depth: int):
        old_depth = self.cluster_dict_[''].max_depth
        for i in range(old_depth, new_depth):
            for name, cluster in self.cluster_dict_.items():
                cluster.max_depth = i + 1

                if len(cluster.name) == i:
                    cluster.update(internals_only=True)
                    cluster.pop()

            self.cluster_dict_ = self._get_cluster_dict()
        return

    def compress(self, metadata_filename: str, integer_filename: str, integer_zip: str):
        metadata = {}

        def get_metadata(node: Cluster):
            metadata[node.name] = [self.data[node.center], node.radius, node.lfd]
            if node.left or node.right:
                get_metadata(node.left)
                get_metadata(node.right)
            return

        get_metadata(self.root)

        with open(metadata_filename, 'wb') as outfile:
            pickle.dump(metadata, outfile)

        if not os.path.exists(integer_filename):
            os.makedirs(integer_filename)

        def write_data(node: Cluster):
            node.compress(integer_filename)
            if node.left or node.right:
                write_data(node.left)
                write_data(node.right)
            return

        write_data(self.root)

        with zipfile.ZipFile(integer_zip, 'w') as outfile:
            [[outfile.write(os.path.join(folder, file),
                            os.path.relpath(os.path.join(folder, file), integer_filename),
                            compress_type=zipfile.ZIP_DEFLATED)
              for file in files]
             for folder, sub_folders, files in os.walk(integer_filename)]

        return