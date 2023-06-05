export NORMAL_TYPE=symm_scale_aware python train_softemlp_checkpoint.py --network=o3mlp --sweep --logoff --bs=256 --epochs=500 --no_early_stop
export NORMAL_TYPE=scale_aware python train_softemlp_checkpoint.py --network=o3mlp --sweep --logoff --bs=256 --epochs=750
export NORMAL_TYPE=symm_aware python train_softemlp_checkpoint.py --network=o3mlp --sweep --logoff --bs=256 --epochs=500 --no_early_stop
