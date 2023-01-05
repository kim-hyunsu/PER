python3 train_softemlp.py --network=o3o2emlp --sweep --ch 180 --trials=4 > motioniclr_0.log
python3 train_softemlp.py --network=o3emlp --sweep --ch 180  --trials=4 > motioniclr_1.log
python3 train_softemlp.py --network=o3mixedemlp --sweep --ch 125  --trials=4 > motioniclr_2.log