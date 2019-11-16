from setuptools import setup

import chess

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='CHESS',
    version=chess.__version__.strip(),
    packages=['chess'],
    url='https://github.com/nishaq503/CHESS',
    license='',
    author=chess.__author__.strip(),
    author_email='',
    description=chess.__doc__.strip(),
    long_description=long_description,
    install_requires=['numpy', 'scipy'],
    python_requires=['>=3.6'],
)
