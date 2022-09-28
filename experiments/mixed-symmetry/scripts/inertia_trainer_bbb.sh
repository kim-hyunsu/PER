#!/bin/bash
for param_loc in 10 5 20
    for lr in 1e-3 6e-4
        do
            python inertia_trainer_bbb.py --network=oxy2oyz2oxz2softemlp --equiv=0,0,0 --wd=1e-10 --gatednonlinearity --trials=2 --ensemble=4 --lr=${lr} --param_loc=${param_loc}
        done