# Projection-based Equivariance Regularizer
Source code of [Regularizing Towards Soft Equivariance Under Mixed Symmetries](https://arxiv.org/abs/2306.00356)

## Environment
```bash
conda create -n per python=3.8
conda activate per
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```

## How to run training for the Synthetic function dataset (inertia)
```bash
cd experiments/mixed-symmetry
mkdir checkpoints
export WANDB_API_KEY=your_wandb_api_key
export SOFT_DATASET=inertia
python train_softemlp_checkpoint.py --network=o3subgroupsoftemlp --auto_equiv
```

## How to run training for Waymo Open Dataset (symmetry-aware normalization)
```bash
cd experiments/mixed-symmetry
mkdir checkpoints
export WANDB_API_KEY=your_wandb_api_key
export SOFT_DATASET=motion
export WAYMO_PATH=/path/to/waymo/open/dataset
export NORMAL_TYPE=symm_aware
python train_softemlp_checkpoint.py --network=o3subgroupsoftemlp --auto_equiv
```

## Citation
```bibtex
@inproceedings{kim2023regularizing,
  title={Regularizing Towards Soft Equivariance Under Mixed Symmetries},
  author={Kim, Hyunsu and Lee, Hyungi and Yang, Hongseok and Lee, Juho},
  booktitle={Proceedings of The 37th International Conference on Machine Learning (ICML 2020)},
  year={2023}
}
```