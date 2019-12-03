from collections import Counter

import numpy as np
from Bio import SeqIO


def parse_fasta():
    characters = ['-', 'A', 'C', 'G', 'T', 'N', 'B', 'R', 'K', 'H', 'W', 'M', 'S', 'Y', 'D', 'V']
    characters_dict = {c: i for i, c in enumerate(characters)}
    characters_dict['.'] = characters_dict['-']

    # # IUPAC masking
    characters_dict['R'] = characters_dict['A']  # A or G
    characters_dict['Y'] = characters_dict['C']  # C or T
    characters_dict['S'] = characters_dict['G']  # G or C
    characters_dict['W'] = characters_dict['A']  # A or T
    characters_dict['K'] = characters_dict['G']  # G or T
    characters_dict['M'] = characters_dict['A']  # A or C
    characters_dict['B'] = characters_dict['C']  # C or G or T
    characters_dict['D'] = characters_dict['A']  # A or G or T
    characters_dict['H'] = characters_dict['A']  # A or C or T
    characters_dict['V'] = characters_dict['A']  # A or C or G
    characters_dict['N'] = characters_dict['A']  # any base

    # [print(c, characters_dict[c]) for c in characters]

    seq_len = 7_682
    num_sequences = 1_075_170

    base_path = '/scratch/nishaq/GreenGenes/'
    fasta_file = base_path + 'GreenGenes.fasta'
    data_path = base_path + 'gg_data.memmap'
    queries_path = base_path + 'gg_queries.memmap'

    queries = np.random.choice(a=num_sequences, size=(10_000,), replace=False)
    queries = list(np.sort(queries))

    data_meta_path = base_path + 'gg_data_metadata.csv'
    queries_meta_path = base_path + 'gg_queries_metadata.csv'

    data_memmap = np.memmap(
        data_path,
        dtype=np.int8,
        mode='w+',
        shape=(num_sequences - 10_000, seq_len)
    )
    queries_memmap = np.memmap(
        queries_path,
        dtype=np.int8,
        mode='w+',
        shape=(10_000, seq_len)
    )

    counter = 0
    # lengths, characters = {}, {}
    with open(fasta_file, 'rU') as fasta_handle, \
            open(data_meta_path, 'w') as data_metafile, \
            open(queries_meta_path, 'w') as queries_metafile:
        data_metafile.write('index, fasta_id, fasta_description\n')
        queries_metafile.write('index, fasta_id, fasta_description\n')
        data_counter = queries_counter = 0

        for record in SeqIO.parse(fasta_handle, 'fasta'):
            seq = str(record.seq)
            # if len(seq) in lengths.keys():
            #     lengths[len(seq)] += 1
            # else:
            #     lengths[len(seq)] = 1
            if len(seq) != seq_len:
                continue

            # characters = characters.union(set(list(seq)))
            chars = set(list(seq))
            if not chars <= set(characters):
                continue
            if chars <= {'-', '.'}:
                continue

            seq_int = np.asarray([characters_dict[c] for c in seq], dtype=np.int8)
            counter += 1
            if counter in queries:
                queries_metafile.write(f'{queries_counter}, {record.id}, {record.description}\n')
                queries_memmap[queries_counter] = seq_int
                queries_counter += 1
            else:
                data_metafile.write(f'{data_counter}, {record.id}, {record.description}\n')
                data_memmap[data_counter] = seq_int
                data_counter += 1
            if data_counter % 10_000 == 0:
                data_memmap.flush()
            if counter % 10_000 == 0:
                print(counter)
                # print(counter, set(characters.keys()))
    data_memmap.flush()
    queries_memmap.flush()
    # print(f'number_of_sequences: {counter}')
    # print(f'characters involved: {set(characters.keys())}')
    # [print(f'{c}: f') for c, f in characters.items()]
    # print('lengths involved:')
    # [print(l, n) for l, n in lengths.items()]

    return


if __name__ == "__main__":
    parse_fasta()
