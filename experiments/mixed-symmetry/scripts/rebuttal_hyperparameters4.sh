SOFT_DATASET=cossim python train_softemlp_checkpoint.py --network=so3scalesubgroupsoftemlp --auto_equiv --sweep --no_early_stop --logoff --noise=1 --sym=scale --adjust_exp=4
SOFT_DATASET=cossim python train_softemlp_checkpoint.py --network=so3scalesubgroupsoftemlp --auto_equiv --sweep --no_early_stop --logoff --noise=1 --sym=scale --adjust_exp=5
SOFT_DATASET=cossim python train_softemlp_checkpoint.py --network=so3scalesubgroupsoftemlp --auto_equiv --sweep --no_early_stop --logoff --noise=1 --sym=scale --adjust_equiv_at=300
SOFT_DATASET=cossim python train_softemlp_checkpoint.py --network=so3scalesubgroupsoftemlp --auto_equiv --sweep --no_early_stop --logoff --noise=1 --sym=scale --adjust_equiv_at=700