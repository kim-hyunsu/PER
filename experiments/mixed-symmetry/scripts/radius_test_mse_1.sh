python train_softemlp.py --network=so3scalesubgroupsoftemlp --auto_equiv --sym=scale  --noise=1 --trials=5 --adjust_exp=5 --equiv=0,0.1,0.1
python train_softemlp.py --network=so3scalemixedemlp --sym=scale  --noise=1 --trials=5 --ch 45
python train_softemlp.py --network=so3scaleemlp --sym=scale  --noise=1 --trials=5
python train_softemlp.py --network=so3scalepartialmixedemlp --sym=scale  --noise=1 --trials=5 --ch 45