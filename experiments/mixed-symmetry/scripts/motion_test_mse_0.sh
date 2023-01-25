NORMAL_TYPE=symm_scale_aware python train_softemlp.py --network=o3partialmixedemlp --sweep --ch=269 --axis=2
NORMAL_TYPE=scale_aware python train_softemlp.py --network=o3partialmixedemlp --sweep --ch=269 --axis=2
NORMAL_TYPE=symm_aware python train_softemlp.py --network=o3subgroupsoftemlp --sweep --auto_equiv