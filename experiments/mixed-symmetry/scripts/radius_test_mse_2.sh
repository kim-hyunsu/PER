python train_softemlp.py --network=so3scalesubgroupsoftemlp --auto_equiv --sym=so3-scale  --noise=1 --trials=5 --init_equiv=0.005 --lr=5e-4
python train_softemlp.py --network=so3scalemixedemlp --sym=so3-scale  --noise=1 --trials=5 --ch 45
python train_softemlp.py --network=so3scaleemlp --sym=so3-scale  --noise=1 --trials=5
python train_softemlp.py --network=so3scalepartialmixedemlp --sym=so3-scale  --noise=1 --trials=5 --ch 45