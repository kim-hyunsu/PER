#!/bin/bash
python inertia_trainer.py \
    --network=o3subgroupsoftemlp \
    --gatednonlinearity \
    --trials=1 \
    --equiv=1,100,0,0 \
    --n_transforms=8 \
    --sign=1 \
    --noise=1.2 > per_1dot2noise_1,100.log
python inertia_trainer.py \
    --network=o3subgroupsoftemlp \
    --gatednonlinearity \
    --trials=1 \
    --equiv=1,200,0,0 \
    --n_transforms=8 \
    --sign=1 \
    --noise=1.2 > per_1dot2noise_1,200.log
python inertia_trainer.py \
    --network=o3subgroupsoftemlp \
    --gatednonlinearity \
    --trials=1 \
    --equiv=1,50,0,0 \
    --n_transforms=8 \
    --sign=1 \
    --noise=1.2 > per_1dot2noise_1,50.log