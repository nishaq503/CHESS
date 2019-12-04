from matplotlib import pyplot as plt

from chess import criterion
from chess.datasets import *
from chess.manifold import *


def plot_clusters(data, labels, manifold):
    for g in manifold.graphs:
        print(len(g.components))
        circles = [plt.Circle(tuple(c.center), c.radius, fill=False) for c in g]
        ax = plt.gca()
        [ax.add_artist(circle) for circle in circles]
        ax.scatter(data[:, 0], data[:, 1], c=labels, s=0.1)
        plt.axis('off')
        plt.show()
    return


def plot_components(data, manifold):
    for g in manifold.graphs:
        label_dict = {p: i for i, component in enumerate(g.components) for c in component for p in c.argpoints}
        labels = [label_dict[i] for i in range(data.shape[0])]
        ax = plt.gca()
        ax.scatter(data[:, 0], data[:, 1], c=labels, s=0.1)
        plt.axis('off')
        plt.show()
    return


def main():
    data, labels = bullseye()
    manifold = Manifold(data, 'euclidean')
    manifold.build(criterion.MinRadius(0.15), criterion.MaxDepth(20), criterion.MinPoints(10))
    print(len(manifold.graphs))
    # plot_clusters(data, labels, manifold)
    plot_components(data, manifold)

    return


if __name__ == '__main__':
    main()
