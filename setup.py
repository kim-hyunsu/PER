from setuptools import setup, find_packages
import sys
import os

setup(name="Residual Pathway Priors",
      description="",
      version='0.0',
      author='-',
      author_email='maf820@nyu.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['h5py', 'objax', 'pytest', 'sklearn',  # jax should be installed first and seperately
                        'olive-oil-ml @ git+https://github.com/mfinzi/olive-oil-ml',
                        'emlp @ git+https://github.com/mfinzi/equivariant-MLP', 'optax', 'tqdm>=4.38',
                        'wandb', 'ml-collections', 'tensorboardX', 'flax==0.3.6', 'gym', 'gdown', 'distrax'],
      packages=['rpp'],
      long_description=open('README.md').read()

      )
