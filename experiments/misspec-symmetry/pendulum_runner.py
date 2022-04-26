import sys  # nopep8
sys.path.append("../trainer/")  # nopep8
sys.path.append("../")  # nopep8
from emlp.nn import MLP, EMLP, MLPH, EMLPH
from emlp.groups import SO2eR3, O2eR3, DkeR3, Trivial, SO
from emlp.reps import Scalar
from hamiltonian_dynamics import IntegratedDynamicsTrainer, DoubleSpringPendulum, hnn_trial
from hamiltonian_dynamics import WindyDoubleSpringPendulum, BHamiltonianFlow
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr, FixedNumpySeed, FixedPytorchSeed
from utils import LoaderTo
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
from rpp.objax import MixedEMLP, MixedEMLPH, MixedMLPEMLP, MixedMLPEMLPH, MixedGroupEMLP, MixedGroupEMLPH
from datasets import ModifiedInertia
import torch.nn as nn
import numpy as np
import pandas as pd
import jax.numpy as jnp
import objax
import argparse
from tqdm import tqdm
import wandb


def main(args):

    num_epochs = 1000
    ndata = 5000
    seed = 2021

    split = {'train': 500, 'val': .1, 'test': .1}
    bs = 500
    logger = []
    equiv_wd = [float(wd) for wd in args.equiv_wd.split(",")]
    basic_wd = [float(wd) for wd in args.basic_wd.split(",")]
    for trial in range(10):
        watermark = "pendulum_log_" + args.network + "_basic" + \
            str(args.basic_wd) + "_equiv" + \
            str(args.equiv_wd) + "_trial" + str(trial)
        wandb.init(
            project="RPP pendulum_runner",
            name=watermark,
            mode="disabled" if args.logoff else "online"
        )
        wandb.config.update(args)
        dataset = DoubleSpringPendulum
        base_ds = dataset(n_systems=ndata, chunk_len=5)
        datasets = split_dataset(base_ds, splits=split)

        dataloaders = {k: LoaderTo(DataLoader(v, batch_size=min(bs, len(v)), shuffle=(k == 'train'),
                                              num_workers=0, pin_memory=False)) for k, v in datasets.items()}
        trainloader = dataloaders['train']
        testloader = dataloaders['val']

        net_config = {'num_layers': 3, 'ch': 128, 'group': SO(3)}
        if args.network.lower() == "emlp":
            model = EMLPH(base_ds.rep_in, Scalar, **net_config)
        elif args.network.lower() == 'mixedemlp':
            model = MixedEMLPH(base_ds.rep_in, Scalar, **net_config)
        elif args.network.lower() == 'mlp':
            model = MLPH(base_ds.rep_in, Scalar, **net_config)
        elif args.network.lower() == 'mixedgroupo2er3emlp':
            net_config["group"] = (net_config["group"], O2eR3())
            model = MixedGroupEMLPH(base_ds.rep_in, Scalar, **net_config)
        elif args.network.lower() == 'mixedgroupso3emlp':
            net_config["group"] = (net_config["group"], SO(3))
            model = MixedGroupEMLPH(base_ds.rep_in, Scalar, **net_config)
        else:
            raise Exception()

        opt = objax.optimizer.Adam(model.vars())

        lr = 2e-3

        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def mse(minibatch):
            (z0, ts), true_zs = minibatch
            pred_zs = BHamiltonianFlow(model, z0, ts[0])
            return jnp.mean((pred_zs - true_zs)**2)

        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def loss(minibatch):
            """ Standard cross-entropy loss """
            (z0, ts), true_zs = minibatch
            pred_zs = BHamiltonianFlow(model, z0, ts[0])
            mse = jnp.mean((pred_zs - true_zs)**2)

            basic_l2 = sum((v.value ** 2).sum()
                           for k, v in model.vars().items() if k.endswith('w_basic'))
            basic1_l2 = sum((v.value ** 2).sum()
                            for k, v in model.vars().items() if k.endswith('w'))
            equiv_l2 = sum((v.value ** 2).sum()
                           for k, v in model.vars().items() if not k.endswith('w_equiv'))
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
        def train_op(batch, lr):
            g, v = grad_and_val(batch)
            opt(lr=lr, grads=g)
            return v

        for epoch in tqdm(range(num_epochs)):
            train_mse = 0
            for batch in trainloader:
                l = train_op(batch, lr)
                train_mse += l[0]*batch[1].shape[0]
            train_mse /= len(trainloader.dataset)
            test_mse = 0
            for batch in testloader:
                l = mse(batch)
                test_mse += l*batch[1].shape[0]
            test_mse /= len(testloader.dataset)
            wandb.log({"train_mse": train_mse, "test_mse": test_mse})
            # tr_loss_wd = np.mean([train_op(batch, lr)
            #                      for batch in trainloader])

        test_loss = np.mean([mse(batch) for batch in testloader])
        tr_loss = np.mean([mse(batch) for batch in trainloader])
        logger.append([trial, tr_loss, test_loss])
        # if args.network.lower() != "mixedemlp":
        #     fname = "log_basic" + str(args.basic_wd) + "_equiv" + str(args.equiv_wd) +\
        #             "_trial" + str(trial) + ".npz"
        #     objax.io.save_var_collection(
        #         "./saved-outputs/" + fname, model.vars())
        wandb.finish()

    save_df = pd.DataFrame(logger)
    fname = "log_" + args.network + "_basic" + str(args.basic_wd) +\
            "_equiv" + str(args.equiv_wd) + ".pkl"
    save_df.to_pickle("./saved-outputs/" + fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pendulum ablation")
    parser.add_argument(
        "--basic_wd",
        type=str,
        default="1e-5",
        help="basic weight decay",
    )
    parser.add_argument(
        "--equiv_wd",
        type=str,
        default="1e-5",
        help="basic weight decay",
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
