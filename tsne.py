import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config


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


def main():
    data: np.memmap = read_data(filename=config.DATA_FILE,
                                num_rows=config.NUM_ROWS - 10_000,
                                num_dims=config.NUM_DIMS,
                                dtype='float32')

    num_samples = data.shape[0]
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


if __name__ == '__main__':
    main()
