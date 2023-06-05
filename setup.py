from setuptools import setup, find_packages
import sys
import os

# revised from https://github.com/mfinzi/residual-pathway-priors/tree/main

setup(name="Projection-based Regularizers",
      description="",
      version='0.0',
      author='-',
      author_email='kim.hyunsu@kaist.ac.kr',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['h5py', 'objax', 'pytest', 'scikit-learn',  # jax should be installed first and seperately
                        'olive-oil-ml @ git+https://github.com/mfinzi/olive-oil-ml',
                        'emlp @ git+https://github.com/mfinzi/equivariant-MLP', 'optax', 'tqdm>=4.38',
                        'wandb', 'ml-collections', 'tensorboardX', 'flax==0.3.6', 'gym', 'gdown', 'distrax'],
      packages=['models'],
      long_description=open('README.md').read()

      )
