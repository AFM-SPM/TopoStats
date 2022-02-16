from setuptools import setup, find_packages

setup(
    name='topostats',
    version='0.1.0',
    url='https://github.com/AFM-SPM/TopoStats',
    author='TopoStats Team',
    packages=find_packages(),
    install_requires=['matplotlib',
        'numpy',
        'pandas',
        'scikit-image',
        'scipy',
        'seaborn'],
    )