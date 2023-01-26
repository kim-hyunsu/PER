NORMAL_TYPE=symm_scale_aware python train_softemlp_checkpoint.py --network=o3o2emlp --sweep --bs=256
NORMAL_TYPE=symm_scale_aware python train_softemlp_checkpoint.py --network=o3emlp --sweep --bs=256
NORMAL_TYPE=symm_scale_aware python train_softemlp_checkpoint.py --network=o3o2mixedemlp --sweep --bs=256 --ch=269
NORMAL_TYPE=symm_scale_aware python train_softemlp_checkpoint.py --network=o3mixedemlp --sweep --bs=256 --ch=269
NORMAL_TYPE=symm_scale_aware python train_softemlp_checkpoint.py --network=o3partialmixedemlp --sweep --bs=256 --ch=269 --axis=2
NORMAL_TYPE=symm_scale_aware python train_softemlp_checkpoint.py --network=o3subgroupsoftemlp --auto_equiv --sweep