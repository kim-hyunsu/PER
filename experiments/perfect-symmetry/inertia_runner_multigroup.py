from re import S
import os  # nopep8
import sys  # nopep8
sys.path.append(f"{os.getcwd()}/../trainer")  # nopep8
sys.path.append(f"{os.getcwd()}/../")  # nopep8
from oil.utils.utils import cosLr, FixedNumpySeed, FixedPytorchSeed
from hamiltonian_dynamics import IntegratedDynamicsTrainer, DoubleSpringPendulum, hnn_trial
from tqdm import tqdm
import argparse
import objax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import torch.nn as nn
from datasets import Inertia
from rpp.objax import ExtraGroupEMLP, MixedEMLP, MixedEMLPH, MultiEMLP, SoftEMLP
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from utils import LoaderTo
from torch.utils.data import DataLoader
from hamiltonian_dynamics import WindyDoubleSpringPendulum, BHamiltonianFlow
from emlp.nn import MLP, EMLP, MLPH, EMLPH
from emlp.groups import SO2eR3, O2eR3, DkeR3, Trivial, SO, O, Embed
from emlp.reps import Scalar
import torch
import wandb


def main(args):

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
        watermark = "inertia_log_" + args.network + "_rglr" + \
            str(args.rglr_coef) + "_trial" + str(trial)
        wandb.init(
            project="RPP inertia_runner_multigroup",
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

        G = dset.symmetry
        if args.network.lower() == "emlp":
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=3, ch=384)
        elif args.network.lower() == "o2emlp":
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=O2eR3(), num_layers=3, ch=384)
        elif args.network.lower() == 'mixedemlp':
            model = MixedEMLP(dset.rep_in, dset.rep_out,
                              group=G, num_layers=3, ch=384)
        elif args.network.lower() == 'mlp':
            model = MLP(dset.rep_in, dset.rep_out,
                        group=G, num_layers=3, ch=384)
        elif args.network.lower() == "multiemlp":
            model = MultiEMLP(dset.rep_in, dset.rep_out,
                              group=(
                                  Embed(O(2), 3, slice(2)),
                                  Embed(O(2), 3, slice(1, 3))
                              ), num_layers=3, ch=384)
        elif args.network.lower() == "extragroupemlp":
            model = ExtraGroupEMLP(dset.rep_in, dset.rep_out,
                                   group=(
                                       Embed(O(2), 3, slice(2)),
                                       Embed(O(2), 3, slice(1, 3))
                                   ), num_layers=3, ch=384)
        elif args.network.lower() == "softemlp":
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             group=G, num_layers=3, ch=384)
        else:
            raise Exception()

        opt = objax.optimizer.Adam(model.vars())  # ,beta2=.99)

        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def mse(x, y):
            yhat = model(x)
            rglr = 0
            if "extra" in args.network.lower():
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W1 = lyr.w.value.reshape(-1)
                    W2 = lyr.Pw1@lyr.Pw@lyr.w.value.reshape(-1)
                    Wdiff = W1-W2
                    b1 = lyr.b.value
                    b2 = lyr.Pb1@lyr.Pb@lyr.b.value
                    bdiff = b1-b2
                    rglr += 0.5*(Wdiff*(lyr.Pw@Wdiff)).sum() + \
                        0.5*(bdiff*(lyr.Pb@bdiff)).sum()
            elif "soft" in args.network.lower():
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W1 = lyr.w.value.reshape(-1)
                    W2 = lyr.Pw@lyr.w.valye.reshape(-1)
                    b1 = lyr.b.value
                    b2 = lyr.Pb@lyr.b.value
                    rglr += 0.5*((W1-W2)**2).sum() + 0.5*((b1-b2)**2).sum()

            return ((yhat-y)**2).mean(), rglr

        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def loss(x, y):
            """ l2 regularized MSE """
            yhat = model(x)
            mse = ((yhat-y)**2).mean()

            rglr = 0
            if "multi" in args.network.lower():
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W1 = lyr.Pw@lyr.w.value.reshape(-1)
                    W2 = lyr.Pw1@lyr.w1.value.reshape(-1)
                    b1 = lyr.Pb@lyr.b.value
                    b2 = lyr.Pb1@lyr.b1.value
                    rglr += 0.5*((W1-W2)**2).sum() + 0.5*((b1-b2)**2).sum()
            elif "extra" in args.network.lower():
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W1 = lyr.w.value.reshape(-1)
                    W2 = lyr.Pw1@lyr.Pw@lyr.w.value.reshape(-1)
                    Wdiff = W1-W2
                    b1 = lyr.b.value
                    b2 = lyr.Pb1@lyr.Pb@lyr.b.value
                    bdiff = b1-b2
                    rglr += 0.5*(Wdiff*(lyr.Pw@Wdiff)).sum() + \
                        0.5*(bdiff*(lyr.Pb@bdiff)).sum()
            elif "soft" in args.network.lower():
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W1 = lyr.w.value.reshape(-1)
                    W2 = lyr.Pw@lyr.w.valye.reshape(-1)
                    b1 = lyr.b.value
                    b2 = lyr.Pb@lyr.b.value
                    rglr += 0.5*((W1-W2)**2).sum() + 0.5*((b1-b2)**2).sum()

            return mse + args.rglr_coef*rglr

        grad_and_val = objax.GradValues(loss, model.vars())

        @objax.Jit
        @objax.Function.with_vars(model.vars()+opt.vars())
        def train_op(x, y, lr):
            g, v = grad_and_val(x, y)
            opt(lr=lr, grads=g)
            return v

        top_mse = float("inf")
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
                      "Regularization": test_rglr})
            if test_mse < top_mse:
                top_mse = test_mse

            # train_mse = np.mean(
            #     [train_op(jnp.array(x), jnp.array(y), lr) for (x, y) in trainloader])

        train_mse = np.mean([mse(jnp.array(x), jnp.array(y))
                            for (x, y) in trainloader])
        test_mse = np.mean([mse(jnp.array(x), jnp.array(y))
                           for (x, y) in testloader])
        logger.append([trial, train_mse, test_mse])
        # if args.network.lower() != 'mixedemlp':
        #     fname = "mdl_inertia_basic" + str(args.basic_wd) + "_equiv" + str(args.equiv_wd) +\
        #             "_trial" + str(trial) + ".npz"
        #     objax.io.save_var_collection(
        #         "./saved-outputs/" + fname, model.vars())
        wandb.finish()
        mse_list.append(top_mse)

    print(f"Test MSE: {np.mean(mse_list)}±{np.std(mse_list)}")
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
        "--logoff",
        action="store_true"
    )
    parser.add_argument(
        "--rglr_coef",
        type=float,
        default=1.
    )
    args = parser.parse_args()

    main(args)
