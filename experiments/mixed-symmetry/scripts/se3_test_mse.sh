python train_softemlp.py --network=se3subgroupsoftemlp --logoff --auto_equiv --ch 128 --noise=-1 --sym=l1distance > se3iclr_0.log
python train_softemlp.py --network=se3partialmixedemlp --logoff --ch 89 --noise=-1 --sym=l1distance > se3iclr_1.log
python train_softemlp.py --network=se3mixedemlp --logoff --ch 89 --noise=-1 --sym=l1distance > se3iclr_2.log
python train_softemlp.py --network=se3emlp --logoff --ch 128 --noise=-1 --sym=l1distance > se3iclr_3.log
python train_softemlp.py --network=se3subgroupsoftemlp --logoff --auto_equiv --ch 128 --noise=-0.5 --sym=ball > se3iclr_4.log
python train_softemlp.py --network=se3partialmixedemlp --logoff --ch 89 --noise=-0.5 --sym=ball > se3iclr_5.log
python train_softemlp.py --network=se3mixedemlp --logoff --ch 89 --noise=-0.5 --sym=ball > se3iclr_6.log
python train_softemlp.py --network=se3emlp --logoff --ch 128 --noise=-0.5 --sym=ball > se3iclr_7.log
python train_softemlp.py --network=se3subgroupsoftemlp --logoff --auto_equiv --ch 128 --noise=-0.5 --sym=distance > se3iclr_8.log
python train_softemlp.py --network=se3partialmixedemlp --logoff --ch 89 --noise=-0.5 --sym=distance > se3iclr_9.log
python train_softemlp.py --network=se3mixedemlp --logoff --ch 89 --noise=-0.5 --sym=distance > se3iclr_10.log
python train_softemlp.py --network=se3emlp --logoff --ch 128 --noise=-0.5 --sym=distance > se3iclr_11.log