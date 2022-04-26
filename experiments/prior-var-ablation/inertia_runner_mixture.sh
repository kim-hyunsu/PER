#!/bin/bash

candidate1=1e-3,1e-2,1e-1
candidate2=1e-2,1e-1,1.0
# for basic_wd in {1e-3,1e-2,1e-1,1.0,10.,100.}
for basic_wd in ${candidate1} ${candidate2}
do
    for equiv_wd in 1e-2 1e-3 1e-4 1e-5
    do
        python modified_inertia.py --equiv_wd=${equiv_wd} --basic_wd=${basic_wd} --network=MixtureEMLP
    done
done
echo All done