from email.policy import default
import sys  # nopep8
sys.path.append("../trainer/")  # nopep8
sys.path.append("../")  # nopep8
from oil.utils.utils import cosLr, FixedNumpySeed, FixedPytorchSeed
from hamiltonian_dynamics import IntegratedDynamicsTrainer, DoubleSpringPendulum, hnn_trial
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
                       MixedMLPEMLP, SoftEMLP, WeightedEMLP, MixedGroupEMLPv2, MixedGroup2EMLPv2, MultiEMLPv2)
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from utils import LoaderTo
from torch.utils.data import DataLoader
from hamiltonian_dynamics import WindyDoubleSpringPendulum, BHamiltonianFlow
from emlp.nn import MLP, EMLP, MLPH, EMLPH
from emlp.groups import SO2eR3, O2eR3, DkeR3, Trivial, SO, SL, O, Embed
from emlp.reps import Scalar
import wandb


def main(args):

    if "soft" in args.network.lower():
        num_epochs = 1500
    else:
        num_epochs = 1000
    ndata = 1000+2000
    seed = 2021
    torch.manual_seed(seed)
    np.random.seed(seed)

    lr = 3e-3

    bs = 500
    logger = []

    equiv_wd = [float(wd) for wd in args.equiv_wd.split(",")]
    basic_wd = [float(wd) for wd in args.basic_wd.split(",")]

    mse_list = []
    for trial in range(10):
        watermark = "inertia_log_" + args.network + "_basic" + \
            str(args.basic_wd) + "_equiv" + \
            str(args.equiv_wd) + "_trial" + str(trial)
        wandb.init(
            project="RPP modified_inertia",
            name=watermark,
            mode="disabled" if args.logoff else "online"
        )
        wandb.config.update(args)

        if args.noise_std == 0:
            assert args.noisy_input == False
            # Initialize dataset with 1000 examples
            dset = ModifiedInertia(3000)
        elif args.noisy_input:
            dset = NoisyModifiedInertia(3000, noise_std=args.noise_std)
        else:
            # Initialize dataset with 1000 examples
            dset = RandomlyModifiedInertia(
                3000, noise_std=args.noise_std, MOG=args.mog)
        split = {'train': -1, 'val': 1000, 'test': 1000}
        datasets = split_dataset(dset, splits=split)
        dataloaders = {k: LoaderTo(DataLoader(v, batch_size=min(bs, len(v)), shuffle=(k == 'train'),
                                              num_workers=0, pin_memory=False)) for k, v in datasets.items()}
        trainloader = dataloaders['train']
        testloader = dataloaders['test']

        G = dset.symmetry
        if args.network.lower() == "mlp":
            model = MLP(dset.rep_in, dset.rep_out,
                        group=G, num_layers=3, ch=384)
        elif args.network.lower() == "emlp":
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=3, ch=384)
        elif args.network.lower() == 'mixedemlp':
            model = MixedEMLP(dset.rep_in, dset.rep_out,
                              group=G, num_layers=3, ch=384)
        elif args.network.lower() == 'mixedmlpemlp':
            model = MixedMLPEMLP(dset.rep_in, dset.rep_out,
                                 group=G, num_layers=3, ch=384)
        elif args.network.lower() == 'mixedgroupso3emlp':
            Model = MixedGroupEMLP if args.v1 else MixedGroupEMLPv2
            model = Model(dset.rep_in, dset.rep_out,
                          group=(G, SO(3)), num_layers=3, ch=384)
        elif args.network.lower() == 'mixedgroupsl3emlp':
            Model = MixedGroupEMLP if args.v1 else MixedGroupEMLPv2
            model = Model(dset.rep_in, dset.rep_out,
                          group=(G, SL(3)), num_layers=3, ch=384)
        elif args.network.lower() == 'mixedgroupo3emlp':
            Model = MixedGroupEMLP if args.v1 else MixedGroupEMLPv2
            model = Model(dset.rep_in, dset.rep_out,
                          group=(G, O(3)), num_layers=3, ch=384)
        elif args.network.lower() == 'mixedgroupo2emlp':
            Model = MixedGroupEMLP if args.v1 else MixedGroupEMLPv2
            model = Model(dset.rep_in, dset.rep_out,
                          group=(G, O2eR3()), num_layers=3, ch=384)
        elif args.network.lower() == 'mixedgroupo2so3emlp':
            Model = MixedGroupEMLP if args.v1 else MixedGroupEMLPv2
            model = Model(dset.rep_in, dset.rep_out,
                          group=(SO(3), O2eR3()), num_layers=3, ch=384)
        elif args.network.lower() == 'mixedgroupmultio2emlp':
            Model = MixedGroup2EMLP if args.v1 else MixedGroup2EMLPv2
            model = Model(dset.rep_in, dset.rep_out,
                          group=(
                              Embed(O(2), 3, slice(2)),
                              Embed(O(2), 3, slice(0, 3, 2)),
                              Embed(O(2), 3, slice(1, 3))
                          ), num_layers=3, ch=384)
        elif args.network.lower() == 'mixedgrouptrio2emlp':
            Model = MixedGroup2EMLP if args.v1 else MixedGroup2EMLPv2
            model = Model(dset.rep_in, dset.rep_out,
                          group=(
                              Embed(O(2), 3, slice(2)),
                              Embed(O(2), 3, slice(2)),
                              Embed(O(2), 3, slice(2))
                          ), num_layers=3, ch=384)
        elif args.network.lower() == 'weightedemlp':
            model = WeightedEMLP(dset.rep_in, dset.rep_out,
                                 group=G, num_layers=3, ch=384)
        elif args.network.lower() == 'weightedemlp2':
            model = WeightedEMLP(dset.rep_in, dset.rep_out,
                                 group=G, num_layers=3, ch=384, weighted=True)
        elif args.network.lower() == "softemlp":
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             group=G, num_layers=3, ch=384)
        else:
            raise Exception()

        opt = objax.optimizer.Adam(model.vars())  # ,beta2=.99)

        @ objax.Jit
        @ objax.Function.with_vars(model.vars())
        def mse(x, y):
            yhat = model(x)
            rglr = 0
            if "soft" in args.network.lower():
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W1 = lyr.w.value.reshape(-1)
                    W2 = lyr.Pw@lyr.w.value.reshape(-1)
                    b1 = lyr.b.value
                    b2 = lyr.Pb@lyr.b.value
                    rglr += 0.5*((W1-W2)**2).sum() + 0.5*((b1-b2)**2).sum()

            return ((yhat-y)**2).mean(), rglr

        @ objax.Jit
        @ objax.Function.with_vars(model.vars())
        def loss(x, y):
            """ l2 regularized MSE """
            yhat = model(x)
            mse = ((yhat-y)**2).mean()

            rglr = 0
            if "soft" in args.network.lower():
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W1 = lyr.w.value.reshape(-1)
                    W2 = lyr.Pw@lyr.w.value.reshape(-1)
                    b1 = lyr.b.value
                    b2 = lyr.Pb@lyr.b.value
                    rglr1 = 0.5*((W1-W2)**2).sum() + 0.5*((b1-b2)**2).sum()
                    rglr2 = (lyr.w.value**2).sum()
                    rglr += args.rglr_alpha*rglr1 + args.rglr_beta*rglr2
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
                    mse += bwd * basic_l2_list[i]
                for i, ewd in enumerate(equiv_wd):
                    mse += ewd * equiv_l2_list[i]

            return mse + rglr

        grad_and_val = objax.GradValues(loss, model.vars())

        @ objax.Jit
        @ objax.Function.with_vars(model.vars()+opt.vars())
        def train_op(x, y, lr):
            g, v = grad_and_val(x, y)
            opt(lr=lr, grads=g)
            return v

        top_mse = float('inf')
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            train_mse = 0
            for x, y in trainloader:
                l = train_op(jnp.array(x), jnp.array(y), lr)
                train_mse += l[0]*x.shape[0]
            train_mse /= len(trainloader.dataset)
            test_mse = 0
            test_rglr = 0
            for x, y in testloader:
                l, rglr = mse(jnp.array(x), jnp.array(y))
                test_mse += l*x.shape[0]
                test_rglr += rglr*x.shape[0]
            test_mse /= len(testloader.dataset)
            test_rglr /= len(testloader.dataset)
            wandb.log({"train_mse": train_mse, "test_mse": test_mse,
                      "Regularizer": test_rglr})
            if test_mse < top_mse:
                top_mse = test_mse

            # train_mse = np.mean(
            #     [train_op(jnp.array(x), jnp.array(y), lr) for (x, y) in trainloader])

        train_mse = np.mean([mse(jnp.array(x), jnp.array(y))
                             for (x, y) in trainloader])
        test_mse = np.mean([mse(jnp.array(x), jnp.array(y))
                            for (x, y) in testloader])
        logger.append([trial, train_mse, test_mse])
        wandb.finish()
        mse_list.append(top_mse)

    print(f"Test MSE: {np.mean(mse_list)}+-{np.std(mse_list)}")
    save_df = pd.DataFrame(logger)
    fname = "inertia_log_" + args.network + "_basic" + \
        str(args.basic_wd) + "_equiv" + str(args.equiv_wd) + ".pkl"
    os.makedirs("./saved-outputs/", exist_ok=True)
    save_df.to_pickle("./saved-outputs/" + fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="modified inertia ablation")
    parser.add_argument(
        "--basic_wd",
        type=str,
        default="1e2",
        help="basic weight decay",
    )
    parser.add_argument(
        "--equiv_wd",
        type=str,
        default=".001",
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
        "--noise_std",
        type=float,
        default=0
    )
    parser.add_argument(
        "--mog",
        action="store_true"
    )
    parser.add_argument(
        "--noisy_input",
        action="store_true"
    )
    parser.add_argument(
        "--v1",
        action="store_true"
    )
    parser.add_argument(
        "--rglr_alpha",
        type=float,
        default=200
    )
    parser.add_argument(
        "--rglr_beta",
        type=float,
        default=1e-5
    )
    args = parser.parse_args()

    main(args)
