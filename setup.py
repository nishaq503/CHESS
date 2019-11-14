from setuptools import setup

import chess

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='CHESS',
    version=chess.__version__,
    packages=['chess'],
    url='https://github.com/nishaq503/CHESS',
    license='',
    author='Najib Ishaq',
    author_email='',
    description='Clustered Hierarchical Entropy-Scaling Search',
    long_description=long_description
)
