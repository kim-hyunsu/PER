python train_softemlp.py --network=o3subgroupsoftemlp --logoff --auto_equiv --ch 128 --noise=-1 --axis=0 > iclr_26.log
python train_softemlp.py --network=o3partialmixedemlp --logoff --ch 89 --noise=-1 --axis=0 > iclr_27.log
python train_softemlp.py --network=o3mixedemlp --logoff --ch 89 --noise=-1 --axis=0 > iclr_28.log
python train_softemlp.py --network=o3emlp --logoff --ch 128 --noise=-1 --axis=0 > iclr_31.log
# python train_softemlp.py --network=o3subgroupsoftemlp --logoff --auto_equiv --ch=128 --noise=-1 > iclr_20.log
python train_softemlp.py --network=o3partialmixedemlp --logoff --ch 89 --noise=-1 > iclr_21.log
# python train_softemlp.py --network=o3mixedemlp --logoff --ch 89 --noise=-1 > iclr_22.log
# python train_softemlp.py --network=o3emlp --logoff --ch 128 --noise=-1 > iclr_29.log
# python train_softemlp.py --network=o3subgroupsoftemlp --logoff --auto_equiv --ch=128 --noise=-1 --axis=1 > iclr_23.log
python train_softemlp.py --network=o3partialmixedemlp --logoff --ch 89 --noise=-1 --axis=1 > iclr_24.log
# python train_softemlp.py --network=o3mixedemlp --logoff --ch 89 --noise=-1 --axis=1 > iclr_25.log
# python train_softemlp.py --network=o3emlp --logoff --ch 128 --noise=-1 --axis=1 > iclr_30.log