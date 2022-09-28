#!/bin/bash
python inertia_trainer.py \
    --network=mixedemlp \
    --gatednonlinearity \
    --trials=1 \
    --n_transforms=8 \
    --sign=1 > rpp_0.log
python inertia_trainer.py \
    --network=mixedemlp \
    --gatednonlinearity \
    --trials=1 \
    --n_transforms=8 \
    --sign=1 > rpp_1.log
python inertia_trainer.py \
    --network=mixedemlp \
    --gatednonlinearity \
    --trials=1 \
    --n_transforms=8 \
    --sign=1 > rpp_2.log