python train_softemlp_checkpoint.py --network=so3scalescaleemlp --sweep --logoff --noise=1 --sym=scale --bs=500 --lr=5e-4 --no_early_stop
python train_softemlp_checkpoint.py --network=so3scaleso3emlp --sweep --logoff --noise=1 --sym=so3 --bs=500 --lr=5e-4 --no_early_stop
python train_softemlp_checkpoint.py --network=so3scalemlp --sweep --no_early_stop --logoff --noise=0 --sym=so3-scale --bs=500 --lr=5e-4
python train_softemlp_checkpoint.py --network=so3scaleemlp --sweep --no_early_stop --logoff --noise=1 --sym=none --bs=500 --lr=5e-4
python train_softemlp_checkpoint.py --network=so3scalemlp --sweep --no_early_stop --logoff --noise=1 --sym=so3 --bs=500 --lr=5e-4
python train_softemlp_checkpoint.py --network=so3scalemlp --sweep --no_early_stop --logoff --noise=1 --sym=scale --bs=500 --lr=5e-4