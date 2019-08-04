import numpy as np

import config
from src.cluster import Cluster
from src.utils import calculate_distance


class Search:
    """ Implements entropy-scaling search with GPU acceleration. """
    def __init__(
            self,
            data: np.memmap,
            distance_function: str,
            name: str,
            reading: bool = False,
            names_file: str = None,
            info_file: str = None,
    ):
        self.data: np.memmap = data
        self.distance_function: str = distance_function
        self.name: str = name

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

    def linear_search(self, query: np.ndarray, radius: float):
        results = []

        for i in range(0, np.shape(self.data)[0], config.BATCH_SIZE):
            end = min(i + config.BATCH_SIZE, np.shape(self.data)[0])
            batch = np.asarray([self.data[j] for j in range(i, end)])
            distances = calculate_distance(query, batch, self.distance_function)
            results.extend([i + j for j, d in enumerate(distances) if d <= radius])

        return results

    def clustered_search(self, query: np.ndarray, radius: float, max_depth: int):
        return self.root.search(query, radius, max_depth)

    def print_names(self, filename: str):
        with open(filename, 'w') as outfile:
            outfile.write('name,index\n')

            def _names_helper(cluster: Cluster):
                if cluster.left or cluster.right:
                    _names_helper(cluster.left)
                    _names_helper(cluster.right)
                else:
                    [outfile.write(f'{cluster.name},{str(p)}\n') for p in cluster.points]
                    outfile.flush()

            _names_helper(self.root)
        return

    def print_info(self, filename: str):
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

    def read_cluster_tree(self):

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

        def build_dict_tree(name: str):
            if name is name_to_points:
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

        def build_cluster_tree(name: str):
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
                return None

        return build_cluster_tree('')
