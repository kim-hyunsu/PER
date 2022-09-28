python train_softemlp.py --network=o3subgroupsoftemlp --logoff --auto_equiv --ch 128 --noise=0 > iclr_32.log
python train_softemlp.py --network=o3partialmixedemlp --logoff --ch 89 --noise=0 > iclr_33.log
python train_softemlp.py --network=o3mixedemlp --logoff --ch 89 --noise=0 > iclr_34.log
python train_softemlp.py --network=o3emlp --logoff --ch 128 --noise=0 > iclr_35.log
python train_softemlp.py --network=o3subgroupsoftemlp --logoff --auto_equiv --ch 128 --noise=-1 --axis=-1 > iclr_36.log
python train_softemlp.py --network=o3partialmixedemlp --logoff --ch 89 --noise=-1 --axis=-1 > iclr_37.log
python train_softemlp.py --network=o3mixedemlp --logoff --ch 89 --noise=-1 --axis=-1 > iclr_38.log
python train_softemlp.py --network=o3emlp --logoff --ch 128 --noise=-1 --axis=-1 > iclr_39.log