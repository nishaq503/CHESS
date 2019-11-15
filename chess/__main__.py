""" CHESS Main Loop.

Loads important variables into the environment,
then presents a REPL.
"""

from chess import CHESS

if __name__ == '__main__':
    from argparse import ArgumentParser
    import numpy as np

    parser = ArgumentParser('CHESS')
    parser.add_argument('dataset', help='path to a numpy memmap.')
    parser.add_argument('metric', help='distance metric from scipy cdist')
    args = parser.parse_args()

    print("""
    Welcome to the CHESS Interpreter.
    """)

    data = np.memmap(args.dataset)
    chess = CHESS(data, args.metric)

    while True:
        code = input('> ')
        try:
            results = eval(code)
        except Exception as e:
            print(e)
        else:
            print(results)
