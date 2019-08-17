import os
import numpy as np
from src.utils import tf_calculate_distance, numpy_calculate_distance, tf_calculate_pairwise_distances

import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def check_hamming(df: str, batch: bool = False):
    characters = ['S', 'H', 'N', 'Y', 'G', 'V', '-', 'W', 'R', 'K', 'T', 'D', 'M', '.', 'B', 'A', 'C']
    characters_dict = {c: i for i, c in enumerate(characters)}

    counter = 0
    data = []
    with open(config.GREENGENES_FASTA_SMALL, 'r') as infile:
        while True:
            line1 = infile.readline()
            if not line1:
                break
            line2 = infile.readline()
            if not line2:
                break
            counter += 1
            if 'unaligned' in line2:
                continue
            sequence = [characters_dict[c] for c in line2[:-1]]
            if len(sequence) != 7682:
                continue
            data.append(sequence)
            if counter >= 10:
                break

    data = np.asarray(data, dtype=np.int8)
    if batch:
        distances = numpy_calculate_distance(data, data, df)
    else:
        seq = data[0]
        rest = data[1:]
        distances = numpy_calculate_distance(seq, rest, df)

    print(distances)

    return


if __name__ == '__main__':
    # check_df('l2')
    check_hamming('hamming')
    # check_hamming('hamming', batch=True)
