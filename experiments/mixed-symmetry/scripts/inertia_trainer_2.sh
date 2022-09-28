#!/bin/bash
python inertia_trainer.py \
    --network=o3subgroupsoftemlp \
    --gatednonlinearity \
    --trials=1 \
    --equiv=0,200,200,200 \
    --n_transforms=8 \
    --sign=1 > per_200,200,200.log
python inertia_trainer.py \
    --network=o3subgroupsoftemlp \
    --gatednonlinearity \
    --trials=1 \
    --equiv=0,100,0.001,0.001 \
    --n_transforms=8 \
    --sign=1 > per_100,3dot1,3dot1.log
python inertia_trainer.py \
    --network=o3subgroupsoftemlp \
    --gatednonlinearity \
    --trials=1 \
    --equiv=0,200,0.001,0.001 \
    --n_transforms=8 \
    --sign=1 > per_200,3dot1,3dot1.log