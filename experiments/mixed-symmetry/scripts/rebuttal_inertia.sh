python train_softemlp_checkpoint.py --network=o3o2emlp --sweep --no_early_stop --logoff --axis=2 --ch 128 --noise=-1
python train_softemlp_checkpoint.py --network=o3o2emlp --sweep --no_early_stop --logoff --axis=1 --ch 128 --noise=-1
python train_softemlp_checkpoint.py --network=o3o2emlp --sweep --no_early_stop --logoff --axis=0 --ch 128 --noise=-1
python train_softemlp_checkpoint.py --network=o3mlp --sweep --no_early_stop --logoff --axis=2 --ch 128 --noise=-1
python train_softemlp_checkpoint.py --network=o3mlp --sweep --no_early_stop --logoff --axis=1 --ch 128 --noise=-1
python train_softemlp_checkpoint.py --network=o3mlp --sweep --no_early_stop --logoff --axis=0 --ch 128 --noise=-1
python train_softemlp_checkpoint.py --network=o3mlp --sweep --no_early_stop --logoff --axis=2 --ch 128 --noise=0
python train_softemlp_checkpoint.py --network=o3mlp --sweep --no_early_stop --logoff --axis=-1 --ch 128 --noise=-1