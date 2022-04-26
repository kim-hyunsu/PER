from email.policy import default
import sys  # nopep8
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
                       MixedMLPEMLP, SoftEMLP, SoftMixedEMLP, SoftMultiEMLP, WeightedEMLP, MixedGroupEMLPv2, MixedGroup2EMLPv2, MultiEMLPv2)
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from utils import LoaderTo
from torch.utils.data import DataLoader
from emlp.nn import MLP, EMLP
from emlp.groups import SO2eR3, O2eR3, DkeR3, Trivial, SO, SL, O, Embed
from rpp.groups import Union
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

    lr = 3e-3

    bs = 500
    logger = []

    equiv_wd = [float(wd) for wd in args.equiv_wd.split(",")]
    basic_wd = [float(wd) for wd in args.basic_wd.split(",")]
    equiv_coef = [float(eq) for eq in args.equiv.split(",")]
    intervals = [int(interval) for interval in args.intervals.split(",")]

    mse_list = []
    for trial in range(args.trials):
        if "soft" in args.network.lower():
            watermark = "{}_eq{}_wd{}_t{}".format(
                args.network, args.equiv, args.wd, trial)
        else:
            watermark = "{}_eq{}_bs{}_{}".format(
                args.network, args.equiv_wd, args.basic_wd, trial)

        wandb.init(
            project="Mixed Symmetry, Inertia",
            name=watermark,
            mode="disabled" if args.logoff else "online"
        )
        wandb.config.update(args)

        # Initialize dataset with 1000 examples
        if args.noise_std == 0:
            dset = ModifiedInertia(3000)
        else:
            dset = RandomlyModifiedInertia(3000, noise_std=args.noise_std)
        split = {'train': -1, 'val': 1000, 'test': 1000}
        datasets = split_dataset(dset, splits=split)
        dataloaders = {k: LoaderTo(DataLoader(v, batch_size=min(bs, len(v)), shuffle=(k == 'train'),
                                              num_workers=0, pin_memory=False)) for k, v in datasets.items()}
        trainloader = dataloaders['train']
        validloader = dataloaders['val']
        testloader = dataloaders['test']

        G = dset.symmetry  # O(3)
        if args.network.lower() == "mlp":
            model = MLP(dset.rep_in, dset.rep_out,
                        group=G, num_layers=3, ch=args.ch)
        elif args.network.lower() == "emlp":
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=3, ch=args.ch)
        elif args.network.lower() == 'mixedemlp':
            model = MixedEMLP(dset.rep_in, dset.rep_out,
                              group=G, num_layers=3, ch=args.ch, gnl=args.gatednonlinearity)
        elif args.network.lower() == 'o2mixedemlp':
            G = Oxy2
            model = MixedEMLP(dset.rep_in, dset.rep_out,
                              group=G, num_layers=3, ch=args.ch)
        elif args.network.lower() == 'tripleo2mixedemlp':
            G = Union(Oxy2, Oyz2, Oxz2)
            model = MixedEMLP(dset.rep_in, dset.rep_out,
                              group=G, num_layers=3, ch=args.ch)
        elif args.network.lower() == "o3softemlp":
            G = (O(3),)
            model = SoftMultiEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init)
        elif args.network.lower() == "o2o3softemlp":
            G = (Oxy2, O(3))
            model = SoftMultiEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init)
        elif args.network.lower() == "tripleo2softemlp":
            G = (Oxy2, Oyz2, Oxz2, O(3))
            model = SoftMultiEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init)
        elif args.network.lower() == "o2o3softmixedemlp":
            G = (Oxy2, O(3))
            model = SoftMixedEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init)
        else:
            raise Exception()

        if args.aug:
            assert not isinstance(G, tuple)
            model = dset.default_aug(model)

        opt = objax.optimizer.Adam(model.vars())  # ,beta2=.99)

        def equiv_regularizer(model, net_name):
            rglr1_list = [0 for _ in range(len(equiv_coef))]
            rglr2 = 0
            if "softemlp" in net_name:
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W = lyr.w.value.reshape(-1)
                    b = lyr.b.value
                    for i, (Pw, Pb) in enumerate(zip(lyr.Pw_list, lyr.Pb_list)):
                        W1 = Pw@W
                        b1 = Pb@b
                        rglr1_list[i] += 0.5 * \
                            ((W-W1)**2).sum() + 0.5*((b-b1)**2).sum()
                    rglr2 += 0.5*(W**2).sum() + 0.5*(b**2).sum()
            elif "softmixedemlp" in net_name:
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W = lyr.w.value.reshape(-1)
                    b = lyr.b.value
                    for i, (Pw1, Pb1) in enumerate(zip(lyr.Pw_list, lyr.Pb_list)):
                        W1 = Pw1@lyr.Pw@W
                        Wdiff = W-W1
                        b1 = Pb1@lyr.Pb@b
                        bdiff = b-b1
                        rglr1_list[i] += 0.5*(Wdiff*(lyr.Pw@Wdiff)).sum() + \
                            0.5*(bdiff*(lyr.Pb@bdiff)).sum()
                    rglr2 += 0.5*(W**2).sum() + 0.5*(b**2).sum()

            return rglr1_list, rglr2

        @ objax.Jit
        @ objax.Function.with_vars(model.vars())
        def mse(x, y):
            yhat = model(x)
            if "soft" in args.network.lower():
                rglr1_list, rglr2 = equiv_regularizer(
                    model, args.network.lower())

            return ((yhat-y)**2).mean(), rglr1_list, rglr2

        @ partial(objax.Jit, static_argnums=(2,))
        @ objax.Function.with_vars(model.vars())
        def loss(x, y, epoch):
            """ l2 regularized MSE """
            yhat = model(x)
            mse = ((yhat-y)**2).mean()

            begin = [0]+intervals[:-1]
            end = intervals
            interval_idx = len(equiv_coef)
            for i, (b, e) in enumerate(zip(begin, end)):
                if b <= epoch < e+1:
                    interval_idx = i

            rglr = 0
            if "soft" in args.network.lower():
                rglr1_list, rglr2 = equiv_regularizer(
                    model, args.network.lower())
                for eq, rglr1 in zip(equiv_coef[:interval_idx], rglr1_list[:interval_idx]):
                    rglr += eq*rglr1
                rglr += args.wd*rglr2
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

        @ partial(objax.Jit, static_argnums=(3,))
        @ objax.Function.with_vars(model.vars()+opt.vars())
        def train_op(x, y, lr, epoch):
            g, v = grad_and_val(x, y, epoch)
            opt(lr=lr, grads=g)
            return v

        top_mse = float('inf')
        top_test_mse = float('inf')
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            train_mse = 0
            for x, y in trainloader:
                l = train_op(jnp.array(x), jnp.array(y), lr, epoch)
                train_mse += l[0]*x.shape[0]
            train_mse /= len(trainloader.dataset)
            valid_mse = 0
            valid_rglr1_list = [0 for _ in range(len(equiv_coef))]
            valid_rglr2 = 0
            for x, y in validloader:
                l, rglr1_list, rglr2 = mse(jnp.array(x), jnp.array(y))
                valid_mse += l*x.shape[0]
                valid_rglr1_list = rglr1_list
                valid_rglr2 = rglr2
            valid_mse /= len(validloader.dataset)
            contents = {"train_mse": train_mse, "valid_mse": valid_mse}
            for i, rglr1 in enumerate(valid_rglr1_list):
                contents[f"equiv_rglr_{i}"] = rglr1
            contents["l2_rglr"] = valid_rglr2
            wandb.log(contents)
            if valid_mse < top_mse:
                top_mse = valid_mse
                test_mse = 0
                for x, y in testloader:
                    l, _, _ = mse(jnp.array(x), jnp.array(y))
                    test_mse += l*x.shape[0]
                test_mse /= len(testloader.dataset)
                top_test_mse = test_mse

        wandb.finish()
        mse_list.append(top_test_mse)

    print(mse_list)
    print(f"Test MSE: {np.mean(mse_list)}±{np.std(mse_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="modified inertia ablation")
    parser.add_argument(
        "--basic_wd",
        type=str,
        default="2",
        help="basic weight decay",
    )
    parser.add_argument(
        "--equiv_wd",
        type=str,
        default="2e-5",
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
        default="200"
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-5
    )
    parser.add_argument(
        "--gated_wd",
        type=float,
        default=2.  # same with basic_wd
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
    args = parser.parse_args()

    main(args)
