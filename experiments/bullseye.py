from matplotlib import pyplot as plt

from chess import criterion
from chess.datasets import *
from chess.manifold import *

MIN_RADIUS = 0.5


def plot_clusters(data, labels, manifold):
    for d, g in enumerate(manifold.graphs):
        circles = [plt.Circle(tuple(c.medoid), c.radius, fill=False) for c in g]
        ax = plt.gca()
        [ax.add_artist(circle) for circle in circles]
        ax.scatter(data[:, 0], data[:, 1], c=labels, s=0.1)
        plt.axis('off')
        plt.title(f'depth {d}, num_components {len(g.components)}')
        plt.show()
    return


def plot_components(data, manifold):
    for d, g in enumerate(manifold.graphs):
        sorted_components = sorted(g.components, key=len)
        print(d, [len(c) for c in sorted_components])

        label_dict = {p: (i if len(component) > 5 else 0) for i, component in enumerate(sorted_components) for c in component for p in c.argpoints}
        print(d, len(set(label_dict.values())))
        labels = [len(sorted_components) - label_dict[i] for i in range(data.shape[0])]
        # circles = [plt.Circle(tuple(c.medoid), (c.radius if c.radius > MIN_RADIUS else 2 * MIN_RADIUS), fill=(c.radius <= MIN_RADIUS)) for c in g]
        circles = [plt.Circle(tuple(c.medoid), c.radius, fill=False) for c in g]
        ax = plt.gca()
        [ax.add_artist(circle) for circle in circles]
        ax.scatter(data[:, 0], data[:, 1], c=labels, s=0.1, cmap='Dark2')
        plt.axis('off')
        plt.title(f'depth {d}, num_components {len(sorted_components)}')
        plt.show()
    return


def main():
    data, labels = bullseye()
    manifold = Manifold(data, 'euclidean')
    manifold.build(criterion.MinRadius(MIN_RADIUS), criterion.MaxDepth(12), criterion.MinPoints(1))
    # plot_clusters(data, labels, manifold)
    plot_components(data, manifold)

    return


if __name__ == '__main__':
    main()
