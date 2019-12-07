import numpy as np

from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D


def get_apogee_embedding():
    apo100k_umap_embedding_memmap = np.memmap(
        filename='/data/nishaq/APOGEE2/apo100k_umap.memmap',
        dtype=np.float32,
        mode='r',
        shape=(100_000, 3),
    )

    apo100k_umap_embedding: np.ndarray = np.zeros_like(apo100k_umap_embedding_memmap)
    apo100k_umap_embedding[:] = apo100k_umap_embedding_memmap[:]
    return apo100k_umap_embedding


def get_greengenes_embedding():
    gg100k_umap_embedding_memmap = np.memmap(
        filename='/data/nishaq/GreenGenes/gg100k_umap.memmap',
        dtype=np.float32,
        mode='r',
        shape=(100_000, 3),
    )

    gg100k_umap_embedding: np.ndarray = np.zeros_like(gg100k_umap_embedding_memmap)
    gg100k_umap_embedding[:] = gg100k_umap_embedding_memmap[:]
    return gg100k_umap_embedding


def plot(angles, data, labels, folder, figsize, dpi, s):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    x_limits = [int(np.min(x)), int(np.max(x))]
    y_limits = [int(np.min(y)), int(np.max(y))]
    z_limits = [int(np.min(z)), int(np.max(z))]
    plt.clf()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=labels, s=s, cmap='Set1')
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)
    # for azimuth in range(angles[0], angles[1]):
    #     ax.view_init(elev=10, azim=azimuth)
    #     plt.savefig(folder + f'{azimuth}.png', bbox_inches='tight', pad_inches=0)
    #     return
    for azimuth in range(100 * angles[0], 100 * angles[1], 25):
        ax.view_init(elev=10, azim=azimuth / 100)
        plt.savefig(folder + f'{azimuth}.png', bbox_inches='tight', pad_inches=0)
        # return
    return


def full_rotation(data, labels, step, folder):
    for i in range(0, 360, step):
        if labels is None:
            labels = make_labels(360 // step)
        plot(
            angles=(i, i + step),
            data=data,
            labels=labels,
            folder=folder,
            figsize=(15, 15),
            dpi=200,
            s=0.025
        )
        # return
    return


def make_labels(label):
    labels = [label for _ in range(100_000)]
    labels[0] = 0  # red, query
    labels[1] = 1  # blue
    labels[2] = 2  # green, potential points
    labels[3] = 3  # purple
    labels[4] = 4  # orange
    labels[5] = 5  # yellow
    labels[6] = 6  # brown
    labels[7] = 7  # pink
    labels[8] = 8  # gray, background
    return labels


def make_apogee_plots():
    folder = f'../presentation/apogee2/umap/'
    data = get_apogee_embedding()
    full_rotation(data=data, labels=None, step=60, folder=folder)
    return


def make_greengenes_plots():
    folder = f'../presentation/greengenes/umap/'
    data = get_greengenes_embedding()

    full_rotation(data=data, labels=None, step=60, folder=folder)
    return


if __name__ == '__main__':
    # print('ready to make plots!')
    make_greengenes_plots()
    make_apogee_plots()
