NORMAL_TYPE=symm_aware python train_softemlp.py --network=o3o2emlp --sweep
NORMAL_TYPE=symm_scale_aware python train_softemlp.py --network=o3subgroupsoftemlp --sweep --auto_equiv
NORMAL_TYPE=scale_aware python train_softemlp.py --network=o3subgroupsoftemlp --sweep --auto_equiv