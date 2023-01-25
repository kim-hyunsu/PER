# Projection-based Equivariance Regularizer

# How to run training for Waymo Open Dataset
```bash
pip install -e .
cd experiments/mixed-symmetry
export WANDB_API_KEY=your_wandb_api_key
export SOFT_DATASET=motion
export WAYMO_PATH=/path/to/waymo/open/dataset
export NORMAL_TYPE=symm_aware
python train_softemlp_checkpoint.py --network=o3subgroupsoftemlp --auto_equiv
```