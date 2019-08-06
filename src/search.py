from typing import List

import numpy as np

import config
from src.cluster import Cluster
from src.utils import tf_calculate_distance


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
        return

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

    def clustered_search(self, query: np.ndarray, radius: float, search_depth: int, logfile: str) -> List[int]:
        """ Perform clustered search.

        :param query: point around with to search.
        :param radius: search radius to use.
        :param search_depth: maximum depth to which to search.
        :param logfile: .csv to write logs in.
        :return: list of indexes of hits.
        """
        return self.root.search(query, radius, search_depth, logfile)

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
                name_to_info[line[0]] = [int(center), float(radius), float(lfd), bool(is_leaf)]

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
