#!/bin/bash
python train_softemlp.py --network=o3subgroupsoftemlp --auto_equiv --trials=1 --equiv=0,100,100,100 --sweep --n_transforms=16 --error_test_samples=16 --noise=0.3 --axis=-1
python train_softemlp.py --network=o3subgroupsoftemlp --auto_equiv --trials=1 --equiv=0,100,100,100 --sweep --n_transforms=16 --error_test_samples=16 --noise=0.6 --axis=-1
python train_softemlp.py --network=o3subgroupsoftemlp --auto_equiv --trials=1 --equiv=0,100,100,100 --sweep --n_transforms=16 --error_test_samples=16 --noise=0.9 --axis=-1