python train_softemlp.py --network=o3subgroupsoftemlp --logoff --auto_equiv --ch 128 --noise=-0.3 --axis=-1 > iclr_40.log
python train_softemlp.py --network=o3partialmixedemlp --logoff --ch 89 --noise=-0.3 --axis=-1 > iclr_41.log
python train_softemlp.py --network=o3mixedemlp --logoff --ch 89 --noise=-0.3 --axis=-1 > iclr_42.log
python train_softemlp.py --network=o3emlp --logoff --ch 128 --noise=-0.3 --axis=-1 > iclr_43.log