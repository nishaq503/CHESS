import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import imageio
from itertools import cycle
import os



def read_data(filename: str, num_rows: int, num_dims: int, dtype) -> np.memmap:
    """ Read data from memmap on disk.

    :param filename: filename to read.
    :param num_rows: number of rows in memmap.
    :param num_dims: number of columns in memmap.
    :param dtype: data type of memmap.
    :return: numpy.memmap object.
    """
    return np.memmap(
        filename=filename,
        dtype=dtype,
        mode='r',
        shape=(num_rows, num_dims),
    )


def write_tsne_vector():
    data: np.memmap = read_data(filename=config.DATA_FILE,
                                num_rows=config.NUM_ROWS - 10_000,
                                num_dims=config.NUM_DIMS,
                                dtype='float32')

    # num_samples = data.shape[0]
    num_samples = 1000
    data_subset = data[:num_samples].copy()

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)
    print('Cumulative explained variation for 50 principal components: {}'
          .format(np.sum(pca_50.explained_variance_ratio_)))

    time_start = time.time()
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    embeddings_df = pd.DataFrame(np.asarray(tsne_pca_results))
    embeddings_df = embeddings_df.rename(columns={i: f'tsne_{i}' for i in range(3)})
    embeddings_df.to_csv('data/tsne_astro.csv', index=False)
    return


def get_sibling(left):
    right = left[:-1]
    final = left[-1]
    final = '1' if final == '2' else '2'
    right += final
    return right


def compare_name(left, right):
    if len(left) == len(right):
        return left == right
    elif len(left) > len(right):
        return left[:len(right)] == right
    else:
        return right[:len(left)] == left


def get_color_list(df, cluster_name, new_color, colors):
    colors = [new_color if compare_name(cluster_name, row['name']) else colors[i]
              for i, row in df.iterrows()]
    return colors


def plot_points(df, colors, fig_name, azimuth_start):
    d1 = df['tsne_0'].values
    d2 = df['tsne_1'].values
    d3 = df['tsne_2'].values
    plt.clf()
    fig = plt.figure(figsize=(100, 100))
    ax = fig.gca(projection='3d')

    ax.scatter(d1, d2, d3, c=colors)
    plt.axis('off')

    for azimuth in range(azimuth_start, azimuth_start + 30, 5):
        ax.view_init(elev=10., azim=azimuth)
        plt.savefig(f'data/astro_movie/{fig_name}_{azimuth}d.png')
    return


def next_azimuth(current_azimuth):
    return (current_azimuth + 30) % 360


def make_astro_gif():
    astro_tsne = pd.read_csv('data/tsne_astro.csv',
                             dtype={'tsne_0': float,
                                    'tsne_1': float,
                                    'tsne_2': float,
                                    'name': str, })

    leaf_names = list(set(list(astro_tsne['name'].values)))
    np.random.shuffle(leaf_names)
    leaf_subset = leaf_names[:100]
    astro_tsne['color'] = 0

    idx = 0
    azimuth = 0
    for i, leaf in enumerate(leaf_subset):
        c = 1
        colors = [0 for _ in range(astro_tsne.shape[0])]
        colors = get_color_list(astro_tsne, leaf, c, colors)
        c += 1
        plot_points(astro_tsne, colors, f'{idx}', azimuth)
        azimuth = next_azimuth(azimuth)
        idx += 1
        leaf_name = leaf
        while leaf_name != '':
            sibling_name = get_sibling(leaf_name)
            colors = get_color_list(astro_tsne, sibling_name, c, colors)
            c += 1
            plot_points(astro_tsne, colors, f'{idx}', azimuth)
            azimuth = next_azimuth(azimuth)
            idx += 1
            parent_name = leaf_name[:-1]
            colors = get_color_list(astro_tsne, sibling_name, c, colors)
            c += 1
            plot_points(astro_tsne, colors, f'{idx}', azimuth)
            azimuth = next_azimuth(azimuth)
            idx += 1
            leaf_name = parent_name

    image_names = [f'data/astro_movie/{i}_{azimuth}d.png'
                   for i in range(idx)
                   for azimuth in range(0, 360, 5)]

    image_0 = imageio.imread(image_names[0])
    im_len = np.shape(image_0)[0]
    quarter = im_len // 4

    image_0 = image_0[quarter: 3 * quarter, quarter + 50: 3 * quarter + 50, :]
    imageio.imwrite('data/astro_movie/new_0.png', image_0)

    names = os.listdir("data/astro_movie/")
    names = [f'data/astro_movie/{name}' for name in names]
    images = [imageio.imread(name) for name in image_names if name in names]
    images_small = [im[quarter: 3 * quarter, quarter + 50: 3 * quarter + 50, :] for im in images]
    imageio.mimsave('astro.gif', images_small)


if __name__ == '__main__':
    make_astro_gif()
