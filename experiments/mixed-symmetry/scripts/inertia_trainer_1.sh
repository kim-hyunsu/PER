#!/bin/bash
python inertia_trainer.py \
    --network=mixedemlp \
    --gatednonlinearity \
    --trials=1 \
    --n_transforms=8 \
    --sign=1 \
    --ch=269 \
    --noise=1.2 > rpp269ch_1dot2noise_0.log
python inertia_trainer.py \
    --network=mixedemlp \
    --gatednonlinearity \
    --trials=1 \
    --n_transforms=8 \
    --sign=1 \
    --ch=269 \
    --noise=1.2 > rpp269ch_1dot2noise_1.log