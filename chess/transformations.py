import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def save_to_memmap(filename: str, data: np.ndarray):
    data_memmap: np.memmap = np.memmap(
        filename=filename,
        dtype=np.float32,
        mode='w+',
        shape=data.shape,
    )
    data_memmap[:] = data[:]
    data_memmap.flush()
    del data_memmap
    return


def transform_apogee(mode: str = 'umap'):
    base_path = '/data/nishaq/APOGEE2/'
    data_file = base_path + 'apo100k_data.memmap'

    num_data = 100_000
    num_dims = 8_575

    data_memmap = np.memmap(
        filename=data_file,
        dtype=np.float32,
        mode='r',
        shape=(num_data, num_dims),
    )

    if mode == 'umap':
        umap_embedding_file = base_path + 'apo100k_umap.memmap'
        umap_embedding = UMAP(n_neighbors=128, n_components=3, metric='euclidean').fit_transform(data_memmap)
        save_to_memmap(umap_embedding_file, umap_embedding)
    elif mode == 'tsne':
        tsne_embedding_file = base_path + 'apo100k_tsne.memmap'
        pca_embedding = PCA(n_components=50).fit_transform(data_memmap)
        tsne_embedding = TSNE(n_components=3, metric='euclidean').fit_transform(pca_embedding)
        save_to_memmap(tsne_embedding_file, tsne_embedding)
    else:
        raise ValueError(f'mode must be either umap or tsne. Got {mode} instead.')

    return


def transform_greengenes(mode: str = 'umap'):
    base_path = '/data/nishaq/GreenGenes/'
    data_file = base_path + 'gg100k_data.memmap'

    num_data = 100_000
    num_dims = 7_682

    data_memmap = np.memmap(
        filename=data_file,
        dtype=np.int8,
        mode='r',
        shape=(num_data, num_dims),
    )

    if mode == 'umap':
        umap_embedding_file = base_path + 'gg100k_umap.memmap'
        umap_embedding = UMAP(n_neighbors=128, n_components=3, metric='hamming').fit_transform(data_memmap)
        save_to_memmap(umap_embedding_file, umap_embedding)
    elif mode == 'tsne':
        tsne_embedding_file = base_path + 'gg100k_tsne.memmap'
        pca_embedding = PCA(n_components=50).fit_transform(data_memmap)
        tsne_embedding = TSNE(n_components=3, metric='hamming').fit_transform(pca_embedding)
        save_to_memmap(tsne_embedding_file, tsne_embedding)
    else:
        raise ValueError(f'mode must be either umap or tsne. Got {mode} instead.')

    return


if __name__ == "__main__":
    print('Ready to transform data!')
    # transform_apogee(mode='umap')
    # transform_greengenes(mode='umap')
    # transform_apogee(mode='tsne')
    # transform_greengenes(mode='tsne')
