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
from datasets import Inertia
from rpp.objax import MixedEMLP, MixedEMLPH, MixedGroupEMLP, MixedMLPEMLP
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from utils import LoaderTo
from torch.utils.data import DataLoader
from hamiltonian_dynamics import WindyDoubleSpringPendulum, BHamiltonianFlow
from emlp.nn import MLP, EMLP, MLPH, EMLPH
from emlp.groups import SO2eR3, O2eR3, DkeR3, Trivial, SO
from rpp.groups import SL
from emlp.reps import Scalar
import wandb


def main(args):

    num_epochs = 300
    ndata = 1000+2000
    seed = 2021

    lr = 3e-3

    bs = 500
    logger = []
    equiv_wd = [float(wd) for wd in args.equiv_wd.split(",")]
    basic_wd = [float(wd) for wd in args.basic_wd.split(",")]

    for trial in range(10):
        watermark = "inertia_log_" + args.network + "_basic" + \
            str(args.basic_wd) + "_equiv" + \
            str(args.equiv_wd) + "_trial" + str(trial)
        wandb.init(
            project="RPP inertia_runner",
            name=watermark,
            mode="disabled" if args.logoff else "online"
        )
        wandb.config.update(args)

        dset = Inertia(3000)  # Initialize dataset with 1000 examples
        split = {'train': -1, 'val': 1000, 'test': 1000}
        datasets = split_dataset(dset, splits=split)
        dataloaders = {k: LoaderTo(DataLoader(v, batch_size=min(bs, len(v)), shuffle=(k == 'train'),
                                              num_workers=0, pin_memory=False)) for k, v in datasets.items()}
        trainloader = dataloaders['train']
        testloader = dataloaders['test']

        G = SL(3)
        if args.network.lower() == "emlp":
            print("Using EMLP")
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=3, ch=384)
        elif args.network.lower() == 'mixedemlp':
            print("Using MixedEMLP")
            model = MixedEMLP(dset.rep_in, dset.rep_out,
                              group=G, num_layers=3, ch=384)
        elif args.network.lower() == 'mlp':
            print("Using MLP")
            model = MLP(dset.rep_in, dset.rep_out,
                        group=G, num_layers=3, ch=384)
        elif args.network.lower() == 'mixedgroupso3emlp':
            print("Using MixedGroupSO3EMLP")
            model = MixedGroupEMLP(dset.rep_in, dset.rep_out, group=(
                G, SO(3)), num_layers=3, ch=384)
        elif args.network.lower() == 'mixedgroupsl3emlp':
            print("Using MixedGroupSL3EMLP")
            model = MixedGroupEMLP(dset.rep_in, dset.rep_out, group=(
                G, SL(3)), num_layers=3, ch=384)
        else:
            raise Exception()

        opt = objax.optimizer.Adam(model.vars())  # ,beta2=.99)

        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def mse(x, y):
            yhat = model(x)
            return ((yhat-y)**2).mean()

        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def loss(x, y):
            """ l2 regularized MSE """
            yhat = model(x)
            mse = ((yhat-y)**2).mean()

            basic_l2 = sum((v.value ** 2).sum()
                           for k, v in model.vars().items() if k.endswith('w_basic'))
            basic1_l2 = sum((v.value ** 2).sum()
                            for k, v in model.vars().items() if k.endswith('w'))
            equiv_l2 = sum((v.value ** 2).sum()
                           for k, v in model.vars().items() if k.endswith('w_equiv'))
            equiv1_l2 = sum((v.value ** 2).sum()
                            for k, v in model.vars().items() if k.endswith('w_equiv1'))
            basic_l2_list = [basic_l2, basic1_l2]
            equiv_l2_list = [equiv_l2, equiv1_l2]

            for i, bwd in enumerate(basic_wd):
                mse += bwd * basic_l2_list[i]
            for i, ewd in enumerate(equiv_wd):
                mse += ewd * equiv_l2_list[i]
            return mse

        grad_and_val = objax.GradValues(loss, model.vars())

        @objax.Jit
        @objax.Function.with_vars(model.vars()+opt.vars())
        def train_op(x, y, lr):
            g, v = grad_and_val(x, y)
            opt(lr=lr, grads=g)
            return v

        for epoch in tqdm(range(num_epochs)):
            train_mse = 0
            for x, y in trainloader:
                l = train_op(jnp.array(x), jnp.array(y), lr)
                train_mse += l[0]*x.shape[0]
            train_mse /= len(trainloader.dataset)
            test_mse = 0
            for x, y in testloader:
                l = mse(jnp.array(x), jnp.array(y))
                test_mse += l*x.shape[0]
            test_mse /= len(testloader.dataset)
            wandb.log({"train_mse": train_mse, "test_mse": test_mse})
            # train_mse = np.mean(
            #     [train_op(jnp.array(x), jnp.array(y), lr) for (x, y) in trainloader])

        train_mse = np.mean([mse(jnp.array(x), jnp.array(y))
                            for (x, y) in trainloader])
        test_mse = np.mean([mse(jnp.array(x), jnp.array(y))
                           for (x, y) in testloader])
        logger.append([trial, train_mse, test_mse])
        # if args.network.lower() == 'mixedemlp':
        #     fname = "mdl_inertia_basic" + str(args.basic_wd) + "_equiv" + str(args.equiv_wd) +\
        #             "_trial" + str(trial) + ".npz"
        #     objax.io.save_var_collection(
        #         "./saved-outputs/" + fname, model.vars())
        wandb.finish()

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
    args = parser.parse_args()

    main(args)
