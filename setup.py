import os

from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as f:
    long_description = f.read()

setup(
    name='-'.join(['CHESS', 'python'] + (['nightly'] if os.getenv('NIGHTLY', None) else [])),
    use_scm_version={
        'local_scheme': lambda version: '',
    },
    setup_requires=['setuptools_scm'],
    packages=['chess'],
    url='https://github.com/nishaq503/CHESS',
    license='',
    author='Najib Ishaq',
    author_email='',
    description='Clustered Hierarchical Entropy-Scaling Search',
    long_description=long_description,
    install_requires=['numpy', 'scipy'],
    python_requires='>=3.6',
)
