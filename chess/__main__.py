""" CHESS Main Loop.

Loads important variables into the environment,
then presents a REPL.
"""

from chess import CHESS, defaults

if __name__ == '__main__':
    from argparse import ArgumentParser
    import numpy as np

    parser = ArgumentParser('CHESS')
    parser.add_argument('dataset', help='path to a numpy memmap.')
    parser.add_argument('metric', help='distance metric from scipy cdist')
    parser.add_argument('--dtype', nargs=1, type=str, help='datatype of the dataset')
    parser.add_argument('--shape', nargs='+', type=int, help='shape of the dataset')
    parser.add_argument('--max-depth', nargs=1, type=int, default=defaults.MAX_DEPTH)
    parser.add_argument('--min-radius', nargs=1, type=defaults.RADII_DTYPE, default=defaults.MIN_RADIUS)
    parser.add_argument('--min-points', type=int, nargs=1, default=defaults.MIN_POINTS)
    args = parser.parse_args()

    print("""
    Welcome to the CHESS Interpreter.
    Building your CHESS object now.
    """)

    data = np.memmap(args.dataset, dtype=args.dtype[0], mode='r', shape=tuple(args.shape))
    chess = CHESS(
        data,
        args.metric,
        max_depth=args.max_depth,
        min_points=args.min_points,
        min_radius=args.min_radius
    )
    chess.build()

    print("""
    Done!
    CHESS object built. Feel free to take a look around.
    """)

    while True:
        code = input('> ')
        try:
            results = eval(code)
        except Exception as e:
            print(e)
        else:
            print(results)
