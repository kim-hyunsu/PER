from email.policy import default
import sys

from experiments.datasets import SyntheticSE3Dataset  # nopep8
sys.path.append("../trainer/")  # nopep8
sys.path.append("../")  # nopep8
import os
from tqdm import tqdm
import argparse
import objax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from datasets import ModifiedInertia, RandomlyModifiedInertia, NoisyModifiedInertia
from rpp.objax import (MixedEMLP, MixedEMLPH, MixedGroup2EMLP, MixedGroupEMLP,
                       MixedMLPEMLP, SoftEMLP, SoftMixedEMLP, SoftMultiEMLP, WeightedEMLP, MixedGroupEMLPv2, MixedGroup2EMLPv2, MultiEMLPv2,
                       MixedEMLPV2)
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from utils import LoaderTo
from torch.utils.data import DataLoader
from rpp.objax import MLP, EMLP
from emlp.groups import SO2eR3, O2eR3, DkeR3, Trivial, SO, SL, O, Embed
from rpp.groups import Union, SE3, TranslationGroup, RotationGroup
import wandb
from functools import partial

Oxy2 = O2eR3()
Oyz2 = Embed(O(2), 3, slice(1, 3))
Oxz2 = Embed(O(2), 3, slice(0, 3, 2))


def main(args):

    num_epochs = args.epochs
    ndata = 1000+2000
    seed = 2022
    torch.manual_seed(seed)
    np.random.seed(seed)

    lr = args.lr

    bs = args.bs
    logger = []

    equiv_wd = [float(wd) for wd in args.equiv_wd.split(",")]
    basic_wd = [float(wd) for wd in args.basic_wd.split(",")]
    equiv_coef = jnp.array([float(eq) for eq in args.equiv.split(",")])
    intervals = [int(interval) for interval in args.intervals.split(",")]

    mse_list = []
    for trial in range(args.trials):
        if "soft" in args.network.lower():
            watermark = "{}_eq{}_wd{}_gt{}_t{}".format(
                args.network, args.equiv, args.wd, args.gated_wd, trial)
        else:
            watermark = "{}_eq{}_bs{}_gt{}_t{}".format(
                args.network, args.equiv_wd, args.basic_wd, args.gated_wd, trial)

        wandb.init(
            project="Mixed Symmetry, SE(3)",
            name=watermark,
            mode="disabled" if args.logoff else "online"
        )
        wandb.config.update(args)

        # Initialize dataset with 1000 examples
        dset = SyntheticSE3Dataset(
            args.num_data, for_mlp=True if args.network.lower() == "mlp" else False,
            noisy=args.noisy, noise=args.noise, complex=args.complex)
        split = {'train': -1, 'val': args.valid_data, 'test': 1000}
        datasets = split_dataset(dset, splits=split)
        dataloaders = {k: LoaderTo(DataLoader(v, batch_size=min(bs, len(v)), shuffle=(k == 'train'),
                                              num_workers=0, pin_memory=False)) for k, v in datasets.items()}
        trainloader = dataloaders['train']
        validloader = dataloaders['val']
        testloader = dataloaders['test']

        G = dset.symmetry  # SE3()
        if args.network.lower() == "mlp":
            G = SO(3)  # just for dimension matching
            model = MLP(dset.rep_in, dset.rep_out,
                        group=G, num_layers=3, ch=args.ch,
                        extend=True)
        elif args.network.lower() == "emlp":
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=3, ch=args.ch,
                         extend=True)
        elif args.network.lower() == "r3emlp":
            G = RotationGroup(3)
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=3, ch=args.ch,
                         extend=True)
        elif args.network.lower() == "t3emlp":
            G = TranslationGroup(3)
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=3, ch=args.ch,
                         extend=True)
        elif args.network.lower() == 'mixedemlp':
            G = (SE3(), RotationGroup(3), TranslationGroup(3))
            model = MixedEMLPV2(dset.rep_in, dset.rep_out,
                                groups=G, num_layers=3, ch=args.ch,
                                gnl=args.gatednonlinearity,
                                extend=True)
        elif args.network.lower() == 'r3mixedemlp':
            G = (RotationGroup(3), RotationGroup(3), TranslationGroup(3))
            model = MixedEMLPV2(dset.rep_in, dset.rep_out,
                                groups=G, num_layers=3, ch=args.ch,
                                extend=True)
        elif args.network.lower() == "r3softemlp":
            G = (RotationGroup(3),)
            model = SoftMultiEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init,
                                  extend=True)
        elif args.network.lower() == "r3t3softemlp":
            G = (RotationGroup(3), TranslationGroup(3))
            model = SoftMultiEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init,
                                  extend=True)
        elif args.network.lower() == "r3t3softmixedemlp":
            G = (RotationGroup(3), TranslationGroup(3))
            model = SoftMixedEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init,
                                  extend=True)
        elif args.network.lower() == "r3se3softmixedemlp":
            G = (RotationGroup(3), SE3())
            model = SoftMixedEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init,
                                  extend=True)
        else:
            raise Exception()

        if args.aug:
            assert not isinstance(G, tuple)
            model = dset.default_aug(model)

        opt = objax.optimizer.Adam(model.vars())  # ,beta2=.99)

        def cosine_schedule(init_value, current_steps, total_steps, alpha=0.0):
            cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * current_steps / total_steps))
            decayed = (1 - alpha) * cosine_decay + alpha
            return init_value * decayed
            
        def equiv_regularizer(model, net_name):
            rglr1_list = [0 for _ in range(len(equiv_coef))]
            rglr2 = 0
            if "softemlp" in net_name:
                print("Using", net_name)
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W = lyr.w.value.reshape(-1)
                    b = lyr.b.value
                    Pw_list = lyr.Pw_list
                    Pb_list = lyr.Pb_list
                    for i, (Pw, Pb) in enumerate(zip(Pw_list, Pb_list)):
                        W1 = Pw@W
                        b1 = Pb@b
                        rglr1_list[i] += 0.5 * \
                            ((W-W1)**2).sum() + 0.5*((b-b1)**2).sum()
                    rglr2 += 0.5*(W**2).sum() + 0.5*(b**2).sum()
            elif "softmixedemlp" in net_name:
                print("Using", net_name)
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W = lyr.Pw@lyr.w.value.reshape(-1)
                    b = lyr.Pb@lyr.b.value
                    Pw_list = lyr.Pw_list
                    Pb_list = lyr.Pb_list
                    for i, (Pw1, Pb1) in enumerate(zip(Pw_list, Pb_list)):
                        W1 = Pw1@W
                        b1 = Pb1@b
                        rglr1_list[i] += 0.5 * \
                            ((W-W1)**2).sum() + 0.5*((b-b1)**2).sum()
                    rglr2 += 0.5*(W**2).sum() + 0.5*(b**2).sum()
            elif "mixedemlp" in net_name:
                print("Using", net_name)
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W = lyr.Pw@lyr.w_equiv.value.reshape(-1) + \
                        lyr.w_basic.value.reshape(-1)
                    b = lyr.Pb@lyr.b.value + lyr.b_basic.value
                    Pw_list = lyr.Pw_list
                    Pb_list = lyr.Pb_list
                    for i, (Pw1, Pb1) in enumerate(zip(Pw_list, Pb_list)):
                        W1 = Pw1@W
                        b1 = Pb1@b
                        rglr1_list[i] += 0.5 * \
                            ((W-W1)**2).sum() + 0.5*((b-b1)**2).sum()
                    rglr2 += 0.5*(W**2).sum() + 0.5*(b**2).sum()

            return rglr1_list, rglr2

        @ objax.Jit
        @ objax.Function.with_vars(model.vars())
        def mse(x, y):
            yhat = model(x)
            rglr1_list = [0.]
            rglr2 = 0.
            if "soft" in args.network.lower():
                rglr1_list, rglr2 = equiv_regularizer(
                    model, args.network.lower())

            return ((yhat-y)**2).mean(), rglr1_list, rglr2, (yhat, y)

        @ objax.Jit
        @ objax.Function.with_vars(model.vars())
        def loss(x, y, equiv):
            """ l2 regularized MSE """
            yhat = model(x)
            mse = ((yhat-y)**2).mean()

            rglr = 0
            if "soft" in args.network.lower():
                rglr1_list, rglr2 = equiv_regularizer(
                    model, args.network.lower())
                for eq, rglr1 in zip(equiv, rglr1_list):
                    rglr += eq*rglr1
                rglr += args.wd*rglr2
            elif "mlp" == args.network.lower():
                basic_l2 = sum((v.value ** 2).sum()
                               for k, v in model.vars().items() if k.endswith('w'))
                rglr += basic_wd[0]*basic_l2
            else:
                basic_l2 = sum((v.value ** 2).sum()
                               for k, v in model.vars().items() if k.endswith('w_basic'))
                equiv_l2 = sum((v.value ** 2).sum()
                               for k, v in model.vars().items() if k.endswith('w_equiv'))
                basic1_l2 = sum((v.value ** 2).sum()
                                for k, v in model.vars().items() if k.endswith('w_basic1'))
                equiv1_l2 = sum((v.value ** 2).sum()
                                for k, v in model.vars().items() if k.endswith('w_equiv1'))
                equiv2_l2 = sum((v.value ** 2).sum()
                                for k, v in model.vars().items() if k.endswith('w_equiv2'))
                basic_l2_list = [basic_l2, basic1_l2]
                equiv_l2_list = [equiv_l2, equiv1_l2, equiv2_l2]

                for i, bwd in enumerate(basic_wd):
                    rglr += bwd * basic_l2_list[i]
                for i, ewd in enumerate(equiv_wd):
                    rglr += ewd * equiv_l2_list[i]
            rglr += args.gated_wd*sum((v.value ** 2).sum()
                                      for k, v in model.vars().items() if k.endswith('w_gated'))

            return mse + rglr

        grad_and_val = objax.GradValues(loss, model.vars())

        @ objax.Jit
        @ objax.Function.with_vars(model.vars()+opt.vars())
        def train_op(x, y, lr, equiv):
            g, v = grad_and_val(x, y, equiv)
            opt(lr=lr, grads=g)
            return v

        top_mse = float('inf')
        top_test_mse = float('inf')
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:

            # begin = [0]+intervals[:-1]
            # end = intervals
            # interval_idx = len(equiv_coef)
            # for i, (b, e) in enumerate(zip(begin, end)):
            #     if b <= epoch < e+1:
            #         interval_idx = i

            # equiv = jnp.zeros_like(equiv_coef)
            # equiv = equiv.at[:interval_idx].set(equiv_coef[:interval_idx])

            begin = [0]+intervals[:-1]
            end = intervals
            interval_idx = 0
            for i, (b, e) in enumerate(zip(begin, end)):
                if b <= epoch < e+1:
                    interval_idx = len(equiv_coef)-i

            equiv = jnp.zeros_like(equiv_coef)
            equiv = equiv.at[interval_idx:].set(equiv_coef[interval_idx:])

            train_mse = 0
            if args.cosine:
                lr_ = cosine_schedule(lr, epoch, num_epochs, alpha=0.0)
            for x, y in trainloader:
                if args.cosine:
                    l = train_op(jnp.array(x), jnp.array(y), lr_, equiv)
                else:
                    l = train_op(jnp.array(x), jnp.array(y), lr, equiv)
                train_mse += l[0]*x.shape[0]
            train_mse /= len(trainloader.dataset)
            valid_mse = 0
            valid_rglr1_list = [0 for _ in range(len(equiv_coef))]
            valid_rglr2 = 0
            for x, y in validloader:
                l, rglr1_list, rglr2, pred = mse(jnp.array(x), jnp.array(y))
                valid_mse += l*x.shape[0]
                valid_rglr1_list = rglr1_list
                valid_rglr2 = rglr2
            valid_mse /= len(validloader.dataset)
            contents = {"train_mse": train_mse, "valid_mse": valid_mse}
            for i, rglr1 in enumerate(valid_rglr1_list):
                contents[f"equiv_rglr_{i}"] = rglr1
            contents["l2_rglr"] = valid_rglr2
            if args.cosine:
                contents["lr"] = lr_
            else:
                contents["lr"] = lr
            wandb.log(contents)
            if valid_mse < top_mse:
                top_mse = valid_mse
                test_mse = 0
                for x, y in testloader:
                    l, _, _, _ = mse(jnp.array(x), jnp.array(y))
                    test_mse += l*x.shape[0]
                test_mse /= len(testloader.dataset)
                top_test_mse = test_mse

        wandb.finish()
        mse_list.append(top_test_mse)
        print(f"Trial {trial+1}, Test MSE: {top_test_mse:.3e}")

    print(f"Total Test MSE: {np.mean(mse_list):.3e}±{np.std(mse_list):.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="modified inertia ablation")
    parser.add_argument(
        "--basic_wd",
        type=str,
        default="1",
        help="basic weight decay",
    )
    parser.add_argument(
        "--equiv_wd",
        type=str,
        default="1e-5",
        help="equiv weight decay",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="MixedEMLP",
        help="type of network {EMLP, MixedEMLP, MLP}",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--logoff",
        action="store_true"
    )
    parser.add_argument(
        "--equiv",
        type=str,
        default="0.5"
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0
    )
    parser.add_argument(
        "--gated_wd",
        type=float,
        default=1  # same with basic_wd
    )
    parser.add_argument(
        "--aug",
        action="store_true"
    )
    parser.add_argument(
        "--gatednonlinearity",
        action="store_true"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1500
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5
    )
    parser.add_argument(
        "--rpp_init",
        action="store_true"
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0
    )
    parser.add_argument(
        "--ch",
        type=int,
        default=384
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default="0,0"  # asummed two groups
    )
    parser.add_argument(
        "--noisy",
        action="store_true"
    )
    parser.add_argument(
        "--complex",
        action="store_true"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=7000
    )
    parser.add_argument(
        "--valid_data",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=500
    )
    parser.add_argument(
        "--cosine",
        action="store_true"
    )
    args = parser.parse_args()

    main(args)
