SOFT_DATASET=inertia python train_softemlp_checkpoint.py --network=o3snsoftemlp --no_early_stop --noise=0  --sweep --logoff --init_equiv=50
SOFT_DATASET=cossim python train_softemlp_checkpoint.py --network=so3scalesnsoftemlp --sweep --logoff --no_early_stop --noise=0 --sym=so3-scale