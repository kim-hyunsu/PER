# Projection-based Equivariance Regularizer

## Environment
```bash
conda create -n per python=3.8
conda activate per
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```

## How to run training for Waymo Open Dataset (example)
```bash
cd experiments/mixed-symmetry
mkdir checkpoints
export WANDB_API_KEY=your_wandb_api_key
export SOFT_DATASET=motion
export WAYMO_PATH=/path/to/waymo/open/dataset
export NORMAL_TYPE=symm_aware
python train_softemlp_checkpoint.py --network=o3subgroupsoftemlp --auto_equiv --logoff
```