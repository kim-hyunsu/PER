#!/bin/bash
python se3_trainer.py \
    --network=mixedemlp \
    --ch 45 \
    --ood_shift=0 \
    --sign=1 \
    --gatednonlinearity \
    --n_transforms=8 \
    --trials=1 \
    --lr=2e-4 \
    --min_lr=1e-5 \
    --sym=t3 > t3_rpp_2e-4to1e-5_0.py
python se3_trainer.py \
    --network=mixedemlp \
    --ch 45 \
    --ood_shift=0 \
    --sign=1 \
    --gatednonlinearity \
    --n_transforms=8 \
    --trials=1 \
    --lr=6e-4 \
    --sym=t3 > t3_rpp_6e-4to0_0.py
