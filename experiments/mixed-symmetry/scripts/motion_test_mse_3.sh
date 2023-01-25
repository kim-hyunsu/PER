NORMAL_TYPE=symm_aware python train_softemlp.py --network=o3emlp --sweep
NORMAL_TYPE=symm_aware python train_softemlp.py --network=o3mixedemlp --sweep --ch=269
NORMAL_TYPE=symm_aware python train_softemlp.py --network=o3partialmixedemlp --sweep --ch=269 --axis=2