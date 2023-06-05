for file in checkpoints/motiono3subgroupsoftemlp*.npz
do
    python train_softemlp_checkpoint.py --network=o3subgroupsoftemlp --model_equiv_error --n_transforms=16 --error_test_samples=16 --checkpoint_path=$file
done