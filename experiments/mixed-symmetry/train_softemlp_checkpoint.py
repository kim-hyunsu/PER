import sys
sys.path.append("../trainer/")  # nopep8
sys.path.append("../")  # nopep8
sys.path.append("../../")  # nopep8
import os
from tqdm import tqdm
import argparse
import objax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from datasets import ModifiedInertia, RandomlyModifiedInertia, SyntheticSE3Dataset, SyntheticRadiusDataset, SyntheticCosSimDataset
from models.objax_per import MLP, ExtendedSoftEMLP, SoftEMLP, RPP, MixedRPP, EMLP, TranslationNormalizer, SoftEMLPBlock
# from emlp.nn.objax import EMLP
from oil.datasetup.datasets import split_dataset
from utils import LoaderTo
from torch.utils.data import DataLoader
from emlp.groups import SO, O, Embed, Scaling, SL, GL, Z, S
import emlp.reps as emlp_reps
from models.groups import Union, SE3, TranslationGroup, RotationGroup, ExtendedEmbed, Reflect
import wandb
from datetime import datetime

from key import WANDB_API_KEY
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

DATASET_CANDIDATES = ["cossim", "se3", "inertia", "motion"]

dataset_name = os.getenv("SOFT_DATASET")
normal_type = os.environ.get("NORMAL_TYPE")
symmetry_type = normal_type
if dataset_name is None:
    raise ValueError("SOFT_DATASET is None")
print(f"$SOFT_DATASET={dataset_name}")
dataset_name = dataset_name.lower()
if dataset_name not in DATASET_CANDIDATES:
    raise ValueError(
        "Invalid Dataset: $SOFT_DATASET={}".format(dataset_name))

if dataset_name == "motion":
    from motion_forecasting import dataset as motion

# Inertia
Oxy2 = Embed(O(2), 3, slice(2))
Oyz2 = Embed(O(2), 3, slice(1, 3))
Oxz2 = Embed(O(2), 3, slice(0, 3, 2))
SLxy2 = Embed(SL(2), 3, slice(2))
SLyz2 = Embed(SL(2), 3, slice(1, 3))
SLxz2 = Embed(SL(2), 3, slice(0, 3, 2))
GLxy2 = Embed(GL(2), 3, slice(2))
GLyz2 = Embed(GL(2), 3, slice(1, 3))
GLxz2 = Embed(GL(2), 3, slice(0, 3, 2))

SO3 = SO(3)
Scale = Scaling(3)
SO3Scale = Union(SO3, Scale)


class ModelSearch():
    def __init__(self, net_name, msebystate, statelength, version):
        self.net_name = net_name
        self.statelength = statelength
        self.modelsearch_cond = None
        self.msebystate = msebystate
        self.top_msebystate_list = [float('inf') for _ in range(statelength)]
        self.version = version

    def begin(self, epoch):
        self.valid_msebystate_list = [0 for _ in range(self.statelength)]
        self.modelsearch_cond = (((epoch+1) % 50 == 0)
                                 and ("hybridsoftemlp" in self.net_name))

    def accumulate(self, x, y):
        if self.modelsearch_cond:
            msebystate_list = self.msebystate(x, y)
            for i, mse_by_state in enumerate(msebystate_list):
                self.valid_msebystate_list[i] += mse_by_state*x.shape[0]

    def end(self, length):
        if self.modelsearch_cond:
            for i in range(self.statelength):
                self.valid_msebystate_list[i] /= length
                if self.top_msebystate_list[i] > self.valid_msebystate_list[i]:
                    self.top_msebystate_list[i] = self.valid_msebystate_list[i]

    def optimal_state(self, current_state):
        if self.modelsearch_cond:
            if self.version == 0:
                ############## version 1 ###############
                optimal = min(range(self.statelength),
                              key=lambda i: self.top_msebystate_list[i])-1
                reset_cond = optimal == -1
                transition_cond = current_state == -1
                if reset_cond or transition_cond:
                    return optimal
            elif self.version == 1:
                ############## version 2 ################
                optimal = min(range(self.statelength),
                              key=lambda i: self.top_msebystate_list[i])-1
                return optimal
            elif self.version == 2:
                ############## version 3 ################
                if current_state == -1:
                    optimal = min(range(self.statelength),
                                  key=lambda i: self.top_msebystate_list[i])-1
                    return optimal
                elif self.top_msebystate_list[0] < self.top_msebystate_list[current_state+1]:
                    return -1
            elif self.version == 3:
                ############## version 4 ################
                optimal = min(range(self.statelength),
                              key=lambda i: self.top_msebystate_list[i])-1
                mse_mean = sum(self.top_msebystate_list) / \
                    len(self.top_msebystate_list)
                mse_var = sum(
                    (ele-mse_mean)**2 for ele in self.top_msebystate_list)/len(self.top_msebystate_list)
                mse_std = jnp.sqrt(mse_var)
                if self.top_msebystate_list[optimal+1] <= mse_mean-0.5*mse_std:
                    return optimal
        return current_state


class AdjustEquiv():
    def __init__(self, initial_equiv):
        self.min_rglr_list = float('inf')*jnp.ones_like(initial_equiv)
        self.warmup = 50
        self.adjustable = False

    def update_rglr(self, _rglr_list):
        if self.warmup > 0:
            self.warmup -= 1
            return
        rglr_list = jnp.array(_rglr_list)
        if len(rglr_list.shape) > 1:
            rglr_list = rglr_list.sum(0)
        cond = jnp.any(rglr_list < self.min_rglr_list)
        self.min_rglr_list = jnp.where(cond, rglr_list, self.min_rglr_list)
        self.adjustable = True

    def adjust(self, equiv, power):
        if not self.adjustable:
            return equiv
        rglr_list = self.min_rglr_list
        max_rglr_list = max(rglr_list)*jnp.ones_like(rglr_list)
        min_rglr = min(jnp.where(equiv == 0, max_rglr_list, rglr_list))
        scale = jnp.power(rglr_list/min_rglr, power)
        return equiv/scale


class RGLR():
    def __init__(self, loss_type, **kwargs):
        self.loss_type = loss_type
        self.epsilon = kwargs["epsilon"]

    def __call__(self, a, b):
        if self.loss_type == "mse":
            return 0.5*((a-b)**2).sum()

        elif self.loss_type == "logmse":
            logdiff = 2*jnp.log(self.epsilon+jnp.abs(a-b))
            logmse = jax.scipy.special.logsumexp(logdiff)
            return logmse

        elif self.loss_type == "csd":
            return cosine_similarity(a, b)


class EarlyStop():
    def __init__(self, patience, begin, disable=False):
        self.patience = patience
        self.wait = 0
        self.best = float("inf")
        self.begin = begin
        self.disable = disable

    def update(self, epoch, new_loss):
        if epoch < self.begin:
            self.wait = 0
        elif new_loss < self.best:
            self.best = new_loss
            self.wait = 0
        else:
            self.wait += 1

    def is_stop(self):
        if self.disable:
            return False
        return self.patience < self.wait


def cosine_similarity(a, b):
    inner = (a*b).sum()
    norm_a = jnp.linalg.norm(a)
    norm_b = jnp.linalg.norm(b)
    return 1-inner/(norm_a*norm_b)


def adjust_equiv(_rglr_list, equiv):
    rglr_list = jnp.array(_rglr_list)
    if len(rglr_list.shape) > 1:
        rglr_list = rglr_list.sum(0)

    mean_regular = (equiv*rglr_list).sum()/(equiv != 0).sum()
    adjusted = mean_regular/rglr_list
    adjusted = jnp.where(equiv == 0, equiv, adjusted)
    return adjusted


def main(args):
    def JIT(*a, **k):
        if args.debug:
            return a[0]
        return objax.Jit(*a, **k)

    args.normal_type = normal_type
    num_epochs = args.epochs
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    lr = args.lr
    bs = args.bs
    data = args.dataset
    if data == "inertia":
        assert "o3" in args.network.lower()
    elif data == "se3":
        assert "se3" in args.network.lower()
    elif data == "cossim":
        assert "so3scale" in args.network.lower()
    elif data == "motion":
        assert "o3" in args.network.lower()
    else:
        raise ValueError("Dataset: {}, Network: {}".format(
            data, args.network
        ))

    if args.model_equiv_error:
        repin = 6*emlp_reps.Vector
        repout = 6*emlp_reps.Vector
        hidden_lyrs = args.layers-1
        if args.network.lower() == "o3subgroupsoftemlp":
            G = (O(3), Oxy2, Oyz2, Oxz2)
            model = SoftEMLP(repin, repout,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
        elif args.network.lower() == "o3o2emlp":
            G = Oxy2  # O(3) for inertia
            model = EMLP(repin, repout,
                         group=G, num_layers=hidden_lyrs, ch=args.ch)
            G = (O(3), Oxy2, Oyz2, Oxz2)
        elif args.network.lower() == "o3misspecifiedsoftemlp":
            G = (O(3), Oxy2, SL(3), SLyz2, GL(3), GLxz2)
            model = SoftEMLP(repin, repout,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
        elif args.network.lower() == "o3misspecified2softemlp":
            G = (O(3), O(3), Oxy2, SL(3), SLyz2, GL(3), GLxz2)
            model = SoftEMLP(repin, repout,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
        elif args.network.lower() == "o3misspecified3softemlp":
            G = (O(3), Oxy2, Oyz2, SLxy2, SLxz2, GLyz2, GLxz2)
            model = SoftEMLP(repin, repout,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
        elif args.network.lower() == "o3misspecified4softemlp":
            G = (O(3), Oxy2, SLyz2, GLxz2)
            model = SoftEMLP(repin, repout,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
        # G = (O(3), O(3), Oxy2, SL(3), SLyz2, GL(3), GLxz2)
        model.groups = G
        equivlength = len(G) if isinstance(G, tuple) else 1
        reference = SoftEMLP(repin, repout,
                             groups=G, num_layers=hidden_lyrs, ch=3,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
        objax.io.load_var_collection(args.checkpoint_path, model.vars())
        rng = np.random.default_rng(seed)
        x = jnp.array(rng.standard_normal(
            (args.error_test_samples, 18)))
        # Group-action=based equivariance error
        error_list = [model.equiv_error(
            i, x, args.n_transforms,
            rep_in_list=reference.rep_in_list,
            rep_out_list=reference.rep_out_list) for i in range(equivlength)]
        print(args.network.lower())
        for i, error in enumerate(error_list):
            print(f"{error}")
            error = np.array(error)
        return
    equiv_wd = [float(wd) for wd in args.equiv_wd.split(",")]
    basic_wd = [float(wd) for wd in args.basic_wd.split(",")]
    intervals = [int(interval) for interval in args.intervals.split(",")]

    mse_list = []
    ood_list = []
    equiv_list = []
    for trial in range(args.trials):
        watermark = "{}_{}noise_{}sign_{}shift_{}_eq{}_wd{}_t{}".format(
            f"axis{args.axis}" if data == "inertia" else args.sym,
            args.noise, args.sign, args.ood_shift, args.network, args.init_equiv, args.wd, trial)

        wandb.init(
            project=f"New Mixed Symmetry, {data}",
            name=watermark,
            mode="disabled" if args.logoff else "online"
        )
        wandb.config.update(args)
        wandb.define_metric("valid_mse", summary="min")

        # O(3) -> Sn
        if data == "inertia":
            dim_swap = True if "o3sn" in args.network.lower() else False
        # SO(3), Scale(3) -> Sn
        elif data == "cossim":
            dim_swap = True if "so3scalesn" in args.network.lower() else False
        else:
            dim_swap = False

        # Initialize dataset with 1000 examples
        if args.noise_std == 0:
            if data == "inertia":
                dset = ModifiedInertia(
                    3000, noise=args.noise, axis=args.axis,
                    sign=args.sign, shift=args.ind_shift, dim_swap=dim_swap)
            elif data == "se3":
                dset = SyntheticSE3Dataset(
                    3000, for_mlp=True if args.network.lower() == "mlp" else False,
                    noisy=args.noisy, noise=args.noise, sym=args.sym, sign=args.sign, shift=args.ind_shift)
            elif data == "cossim":
                dset = SyntheticCosSimDataset(
                    3000, noise=args.noise, sym=args.sym, sign=args.sign, shift=args.ind_shift)
        else:
            if data == "inertia":
                dset = RandomlyModifiedInertia(
                    3000, noise_std=args.noise_std, shift=args.ind_shift,
                    dim_swap=dim_swap)
        if args.sign == 0:
            if data == "inertia":
                ood = ModifiedInertia(
                    1000, noise=args.noise, axis=args.axis,
                    shift=args.ood_shift,dim_swap=dim_swap)
            elif data == "se3":
                ood = SyntheticSE3Dataset(1000,
                                          for_mlp=True if args.network.lower() == "mlp" else False,
                                          noisy=args.noisy, noise=args.noise,
                                          sym=args.sym, shift=args.ood_shift)
            elif data == "cossim":
                ood = SyntheticCosSimDataset(
                    1000, noise=args.noise, sym=args.sym, shift=args.ood_shift)
            else:
                ood = None
        else:
            if data == "inertia":
                ood = ModifiedInertia(
                    1000, noise=args.noise, axis=args.axis,
                    sign=-args.sign, dim_swap=dim_swap)
            elif data == "se3":
                ood = SyntheticSE3Dataset(1000,
                                          for_mlp=True if args.network.lower() == "mlp" else False,
                                          noisy=args.noisy, noise=args.noise,
                                          sym=args.sym, shift=args.ood_shift, sign=-args.sign)
            elif data == "cossim":
                ood = SyntheticCosSimDataset(
                    1000, noise=args.noise, sym=args.sym, shift=args.ood_shift, sign=-args.sign)
            else:
                ood = None
        if data != "motion":
            split = {'train': -1, 'val': 1000, 'test': 1000}
            datasets = split_dataset(dset, splits=split)
            ood_dataset = split_dataset(ood, splits={'ood': -1})["ood"]
            dataloaders = {
                k: LoaderTo(DataLoader(v, batch_size=min(bs, len(v)),
                                       shuffle=(k == 'train'),
                                       num_workers=0, pin_memory=False)) for k, v in datasets.items()
            }
            trainloader = dataloaders['train']
            validloader = dataloaders['val']
            testloader = dataloaders['test']
            oodloader = LoaderTo(DataLoader(ood_dataset, batch_size=min(bs, len(ood_dataset)),
                                            num_workers=0, pin_memory=False))
            lengths = {
                "train": len(trainloader.dataset),
                "valid": len(validloader.dataset),
                "test": len(testloader.dataset),
                "ood": len(oodloader.dataset)
            }
        else:
            # dset = motion.Dataloader("training", batch_size=args.bs, normal_type=args.normal_type)
            dset = motion.Dataloader("training", batch_size=args.bs)
            trainloader = dset
            # validloader = motion.Dataloader("validation", batch_size=args.bs, normal_type=args.normal_type)
            validloader = motion.Dataloader("validation", batch_size=args.bs)
            # testloader = motion.Dataloader("testing", batch_size=args.bs, normal_type=args.normal_type)
            testloader = motion.Dataloader("testing", batch_size=args.bs)
            lengths = {
                "train": len(trainloader),
                "valid": len(validloader),
                "test": len(testloader)
            }

        hidden_lyrs = args.layers-1
        network_name = args.network.lower()
        # MLP
        if network_name == "o3mlp":
            args.basic_wd = [0.]
            G = dset.symmetry  # O(3) for inertia
            model = MLP(dset.rep_in, dset.rep_out,
                        group=G, num_layers=hidden_lyrs, ch=args.ch)
            rng = jax.random.PRNGKey(0)
            input = jax.random.normal(rng, (4, 20))
            _equiv_error = model.equiv_error_alone(input, 4)
            print("equiv error", _equiv_error)
        elif network_name == "se3mlp":
            args.basic_wd = [0.]
            G = SO(3)  # just for dimension matching
            model = MLP(dset.rep_in, dset.rep_out,
                        group=G, num_layers=hidden_lyrs, ch=args.ch,
                        extend=True)
        elif network_name == "so3scalemlp":
            args.basic_wd = [0.]
            G = dset.symmetry
            model = MLP(dset.rep_in, dset.rep_out,
                        group=G, num_layers=hidden_lyrs, ch=args.ch)

        # Frame averaging
        elif network_name == "o3famlp":
            assert args.num_frames > 0
            args.basic_wd = [0.]
            G = SO(3)
            model = MLP(dset.rep_in, dset.rep_out,
                        group=G, num_layers=hidden_lyrs, ch=args.ch,
                        fa = args.num_frames)
            rng = jax.random.PRNGKey(0)
            input = jax.random.normal(rng, (4, 20))
            _equiv_error = model.equiv_error_alone(input, 4)
            print("equiv error", _equiv_error)
        elif network_name == "so3scalefamlp":
            assert args.num_frames > 0
            args.basic_wd = [0.]
            G = dset.symmetry
            model = MLP(dset.rep_in, dset.rep_out,
                        group=G, num_layers=hidden_lyrs, ch=args.ch,
                        fa = args.num_frames)


        elif network_name == "o3o2emlp":
            G = [Oyz2, Oxz2, Oxy2][args.axis]  # O(3) for inertia
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=hidden_lyrs, ch=args.ch)
        elif network_name == "so3scaleso3emlp":
            G = SO3
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=hidden_lyrs, ch=args.ch)
        elif network_name == "so3scalescaleemlp":
            G = Scale
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=hidden_lyrs, ch=args.ch)

        # EMLP
        elif network_name == "o3emlp":
            G = dset.symmetry  # O(3) for inertia
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=hidden_lyrs, ch=args.ch)
            rng = jax.random.PRNGKey(0)
            input = jax.random.normal(rng, (4, 20))
            _equiv_error = model.equiv_error_alone(input, 4)
            print("equiv error", _equiv_error)
        elif network_name == "se3emlp":
            G = dset.symmetry  # SE3() for se3
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=hidden_lyrs, ch=args.ch,
                         extend=True)
        elif network_name == "so3scaleemlp":
            G = dset.symmetry  # Union(SO(3), Scaling(3)) for cossim
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=hidden_lyrs, ch=args.ch)

        # RPP
        elif network_name == "o3o2mixedemlp":
            G = (Oxy2, Oxy2, Oxy2, Oyz2)
            model = RPP(dset.rep_in, dset.rep_out,
                        groups=G, num_layers=hidden_lyrs, ch=args.ch,
                        gnl=args.gatednonlinearity)
        elif network_name == 'o3mixedemlp':
            G = (O(3), Oxy2, Oyz2, Oxz2)
            model = RPP(dset.rep_in, dset.rep_out,
                        groups=G, num_layers=hidden_lyrs, ch=args.ch,
                        gnl=args.gatednonlinearity)
        elif network_name == 'se3mixedemlp':
            G = (SE3(), RotationGroup(3),  TranslationGroup(3))
            model = RPP(dset.rep_in, dset.rep_out,
                        groups=G, num_layers=hidden_lyrs, ch=args.ch,
                        gnl=args.gatednonlinearity,
                        extend=True)
        elif network_name == 'so3scalemixedemlp':
            G = (SO3Scale, SO3, Scale)
            model = RPP(dset.rep_in, dset.rep_out,
                        groups=G, num_layers=hidden_lyrs, ch=args.ch,
                        gnl=args.gatednonlinearity)

        # MixedRPP
        elif network_name == "o3partialmixedemlp":
            G = (O(3), Oxy2, Oyz2, Oxz2)
            model = MixedRPP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity)
            selected_state = (args.axis+1) % 3
            model.set_state(selected_state)
            print("selected state", selected_state)
        elif network_name == 'se3partialmixedemlp':
            G = (SE3(), RotationGroup(3),  TranslationGroup(3))
            model = MixedRPP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             extend=True)
            selected_state = 1
            if args.sym == "ball":
                selected_state = 0
            elif args.sym == "l1distance":
                selected_state = 1
            model.set_state(selected_state)
            print("selected state", selected_state)
        elif network_name == "so3scalepartialmixedemlp":
            G = (SO3Scale, SO3, Scale)
            model = MixedRPP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity)
            selected_state = 0
            if args.sym == "scale":
                selected_state = 1
            model.set_state(selected_state)
            print("selected state", selected_state)

        # HybridSoftEMLP
        elif network_name == "o3hybridsoftemlp":
            G = (O(3), Oxy2, Oyz2, Oxz2)
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
            model.set_state(args.initial_state)  # not necessary
        elif network_name == "se3hybridsoftemlp":
            G = (SE3(), RotationGroup(3), TranslationGroup(3))
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init,
                             extend=True)
            model.set_state(args.initial_state)  # not necessary
        elif network_name == "so3scalehybridsoftemlp":
            G = (SO3Scale, SO3, Scale)
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
            model.set_state(args.initial_state)  # not necessary

        # SoftMultiEMLP
        elif network_name == "o3subgroupsoftemlp":
            G = (O(3), Oxy2, Oyz2, Oxz2)
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init,
                             flex=args.flex_act)
            model.set_state(args.initial_state)  # not necessary
        elif network_name == "se3subgroupsoftemlp":
            G = (SE3(), RotationGroup(3), TranslationGroup(3))
            model = ExtendedSoftEMLP(dset.rep_in, dset.rep_out,
                                     groups=G, num_layers=hidden_lyrs, ch=args.ch,
                                     gnl=args.gatednonlinearity,
                                     rpp_init=args.rpp_init,
                                     extend=True)
            model.set_state(args.initial_state)  # not necessary
        elif network_name == "o3subgroupso3softemlp":
            G = (O(3), SO(3), Reflect(3))
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init,
                             flex=args.flex_act)
            model.set_state(args.initial_state)  # not necessary
        elif network_name == "so3scalesubgroupsoftemlp":
            G = (SO3Scale, SO3, Scale)
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init,
                             flex=args.flex_act)
            model.set_state(args.initial_state)  # not necessary
        elif network_name == "o3misspecifiedsoftemlp":
            G = (O(3), Oxy2, SL(3), SLyz2, GL(3), GLxz2)
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
            model.set_state(args.initial_state)  # not necessary
        elif network_name == "o3misspecified2softemlp":
            G = (O(3), O(3), Oxy2, SL(3), SLyz2, GL(3), GLxz2)
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
            model.set_state(args.initial_state)  # not necessary
        elif network_name == "o3misspecified3softemlp":
            G = (O(3), Oxy2, Oyz2, SLxy2, SLxz2, GLyz2, GLxz2)
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
            model.set_state(args.initial_state)  # not necessary
        elif network_name == "o3misspecified4softemlp":
            G = (O(3), Oxy2, SLyz2, GLxz2)
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
            model.set_state(args.initial_state)  # not necessary
        elif network_name == "o3snsoftemlp":
            G = (S(5), S(5))
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
        elif network_name == "so3scalesnsoftemlp":
            G = (S(3), S(3))
            model = SoftEMLP(dset.rep_in, dset.rep_out,
                             groups=G, num_layers=hidden_lyrs, ch=args.ch,
                             gnl=args.gatednonlinearity,
                             rpp_init=args.rpp_init)
        else:
            raise Exception("Invalid Model Name")

        equivlength = len(G) if isinstance(G, tuple) else 1
        if args.equiv != "":
            equiv_coef = [0. for _ in range(equivlength)]
            for i, eq in enumerate(args.equiv.split(",")):
                if i+1 > equivlength:
                    break
                equiv_coef[i] = float(eq)
        else:
            equiv_coef = [
                0 if i == 0 else args.init_equiv for i in range(equivlength)]
        equiv = jnp.array(equiv_coef)
        threshold_list = [0. for _ in range(equivlength)]
        for i, th in enumerate(args.threshold.split(",")):
            if i+1 > equivlength:
                break
            threshold_list[i] = float(th)
        threshold = jnp.array(threshold_list)

        if args.aug:
            assert not isinstance(G, tuple)
            model = dset.default_aug(model)

        opt = objax.optimizer.Adam(model.vars())  # ,beta2=.99)

        def cosine_schedule(init_value, current_steps, total_steps, alpha=0.0, min_value=0):
            cosine_decay = 0.5 * \
                (1 + jnp.cos(jnp.pi * current_steps / total_steps))
            decayed = (1 - alpha) * cosine_decay + alpha
            return (init_value-min_value) * decayed + min_value

        floss = RGLR(args.loss_type, epsilon=1e-45)

        @JIT
        @objax.Function.with_vars(model.vars())
        def equiv_regularizer(except_bias=False):
            net_name = network_name
            rglr1_list = [[0 for _ in range(equivlength)]
                          for _ in range(args.layers)]
            rglr2 = 0
            rglr3_list = [[0 for _ in range(equivlength)]
                          for _ in range(args.layers)]
            if "softemlp" in net_name:
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W = lyr.w.value.reshape(-1)
                    b = lyr.b.value
                    Pw_list = lyr.Pw_list
                    Pb_list = lyr.Pb_list
                    for j, (Pw, Pb) in enumerate(zip(Pw_list, Pb_list)):
                        # single projection
                        W1 = Pw@W
                        b1 = Pb@b
                        W_mse = floss(W, W1)
                        b_mse = floss(b, b1)
                        l2norm = jnp.where(except_bias, W_mse, W_mse+b_mse)
                        W_cse = cosine_similarity(W, W1)
                        b_cse = cosine_similarity(b, b1)
                        rglr1_list[i][j] += l2norm
                        rglr3_list[i][j] += W_cse+b_cse
                    rglr2 += 0.5*(W**2).sum() + 0.5*(b**2).sum()
            elif "softmixedemlp" in net_name:
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W = lyr.w.value.reshape(-1)
                    b = lyr.b.value
                    Pw_list = lyr.Pw_list
                    Pb_list = lyr.Pb_list
                    for j, (Pw1, Pb1) in enumerate(zip(Pw_list, Pb_list)):
                        # double projection
                        W1 = Pw1@lyr.Pw@W
                        Wdiff = W-W1
                        b1 = Pb1@lyr.Pb@b
                        bdiff = b-b1
                        W_mse = 0.5*(Wdiff*(lyr.Pw@Wdiff)).sum()
                        b_mse = 0.5*(bdiff*(lyr.Pb@bdiff)).sum()
                        l2norm = jnp.where(except_bias, W_mse, W_mse+b_mse)
                        rglr1_list[i][j] += l2norm
                    rglr2 += 0.5*(W**2).sum() + 0.5*(b**2).sum()
            elif "mixedemlp" in net_name:
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    # residual pathway
                    W = lyr.Pw@lyr.w_equiv.value.reshape(-1) + \
                        lyr.w_basic.value.reshape(-1)
                    b = lyr.Pb@lyr.b.value + lyr.b_basic.value
                    Pw_list = lyr.Pw_list
                    Pb_list = lyr.Pb_list
                    for j, (Pw1, Pb1) in enumerate(zip(Pw_list, Pb_list)):
                        W1 = Pw1@W
                        b1 = Pb1@b
                        W_mse = floss(W, W1)
                        b_mse = floss(b, b1)
                        l2norm = jnp.where(except_bias, W_mse, W_mse+b_mse)
                        # Caution: i+1 instead of i
                        rglr1_list[i][j+1] += l2norm
                    rglr2 += 0.5*(W**2).sum() + 0.5*(b**2).sum()

            return rglr1_list, rglr2, rglr3_list

        @JIT
        @objax.Function.with_vars(model.vars())
        def msebystate(x, y):
            net_name = args.network.lower()
            if "hybridsoftemlp" in net_name:
                current_state = model.get_current_state()
                msebystate_list = []
                for state in range(-1, equivlength):
                    model.set_state(state)
                    yhat_prime = model(x)
                    msebystate_list.append(((yhat_prime-y)**2).mean())
                model.set_state(current_state)
            else:
                msebystate_list = [0. for _ in range(equivlength)]

            return msebystate_list

        @JIT
        @objax.Function.with_vars(model.vars())
        def mse(x, y):
            yhat = model(x)
            return ((yhat-y)**2).mean()

        @JIT
        @objax.Function.with_vars(model.vars())
        def ade(x, y):
            yhat = model(x)
            dist = jnp.sqrt(((yhat-y)**2).reshape(-1, 6, 3).sum(-1))
            return dist.mean()

        @JIT
        @objax.Function.with_vars(model.vars())
        def unnormalized_mse(x, y):
            yhat = model(x)
            # mean = jnp.array(motion.NORMAL[args.normal_type]["mean"])
            # std = jnp.array(motion.NORMAL[args.normal_type]["std"])
            mean = jnp.array(motion.MEAN)
            std = jnp.array(motion.STD)
            std = jnp.tile(std, 6)  # trajectory
            std = jnp.expand_dims(std, axis=0)  # batch dimension
            return (((yhat-y)*std)**2).mean()

        @JIT
        @objax.Function.with_vars(model.vars())
        def unnormalized_ade(x, y):
            yhat = model(x)
            mean = jnp.array(motion.MEAN)
            std = jnp.array(motion.STD)
            std = jnp.tile(std, 6)  # trajectory
            std = jnp.expand_dims(std, axis=0)  # batch dimension
            dist = jnp.sqrt((((yhat-y)*std)**2).reshape(-1, 6, 3).sum(-1))
            return dist.mean()

        @JIT
        @objax.Function.with_vars(model.vars())
        def loss(x, y, equiv):
            """ l2 regularized MSE """
            yhat = model(x)
            mse = ((yhat-y)**2).mean()

            rglr = 0
            net_name = args.network.lower()
            if "soft" in net_name:
                rglr1_list, rglr2, _ = equiv_regularizer(
                    except_bias=args.no_bias_regular)
                for rglr1s in rglr1_list:
                    for eq, rglr1 in zip(equiv, rglr1s):
                        rglr += eq*rglr1
                rglr += args.wd*rglr2
            elif net_name == "mlp":
                basic_l2 = sum((v.value ** 2).sum()
                               for k, v in model.vars().items() if k.endswith('w'))
                rglr += basic_wd[0]*basic_l2
            elif "mixedemlp" in net_name:
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

            return args.loss_scale*mse + rglr, mse, rglr

        grad_and_val = objax.GradValues(loss, model.vars())

        @JIT
        @objax.Function.with_vars(model.vars()+opt.vars())
        def train_op(x, y, lr, equiv):
            g, (v, _mse, _rglr) = grad_and_val(x, y, equiv)
            opt(lr=lr, grads=g)
            return v, _mse, _rglr

        rglr1_list, rglr2, rglr3_list = equiv_regularizer(
            except_bias=args.no_bias_regular)
        print(f"Initial Equiv {args.init_equiv} Regularizers")
        rglr1_array = jnp.array(rglr1_list)
        print(rglr1_array.sum(0))

        top_mse = float('inf')
        top_ood_mse = float('inf')
        top_test_mse = float('inf')
        top_test_epoch = -1
        pbar = tqdm(range(num_epochs))
        statelength = len(G)+1 if isinstance(G, tuple) else 2
        net_name = network_name
        ms = ModelSearch(net_name, msebystate, statelength,
                         version=args.modelsearch)
        ae = AdjustEquiv(equiv)
        es = EarlyStop(patience=50, begin=2*args.adjust_equiv_at,
                       disable=args.no_early_stop)
        train_mse = None
        train_loss = None
        valid_mse = None
        test_mse_updated = False
        train_length = None
        valid_length = None
        test_length = None
        checkpoint_path = None

        def reporting(info):
            test_mse_updated = info["test_mse_updated"]
            contents = dict()
            contents["lr"] = np.array(lr)
            contents["state"] = model.get_current_state() if hasattr(
                model, 'get_current_state') else -2
            if args.n_transforms > 0:
                rng = np.random.default_rng(seed)
                x = jnp.array(rng.standard_normal(
                    (args.error_test_samples, dset.dim)))
                if not args.sweep or test_mse_updated:
                    # Group-action=based equivariance error
                    error_list = [model.equiv_error(
                        i, x, args.n_transforms) for i in range(equivlength)]
                    for i, error in enumerate(error_list):
                        error = np.array(error)
                        contents[f"equiv_error_{i}"] = error
                        wandb.summary[f"equiv_error_{i}"] = error

                # True group-action-based equivariance error
                if info["epoch"] == 0:
                    print("epoch", info["epoch"])
                    true_error_list = [model.equiv_error(
                        i, x, args.n_transforms, forward=dset) for i in range(equivlength)]
                    for i, true_error in enumerate(true_error_list):
                        true_error = np.array(true_error)
                        wandb.summary[f"true_error_{i}"] = true_error
            if info["epoch"] == 1:
                wandb.summary["train_length"] = info["train_length"]
                wandb.summary["valid_length"] = info["valid_length"]
            if test_mse_updated:
                wandb.summary["test_length"] = info["test_length"]

            if train_mse is not None:
                contents["train_mse"] = info["train_mse"]
            if train_loss is not None:
                contents["train_loss"] = info["train_loss"]
            if valid_mse is not None:
                contents["valid_mse"] = info["valid_mse"]
            for i, eq in enumerate(equiv):
                eq = np.array(eq)
                contents[f"equiv_coef_{i}"] = eq
            if not args.sweep or test_mse_updated:
                for lyr, rglr1 in enumerate(info["rglr1_list"]):
                    for g, rglr in enumerate(rglr1):
                        # contents[f"equiv_rglr_l{lyr}g{g}"] = rglr
                        if contents.get(f"equiv_rglr_{g}") is None:
                            contents[f"equiv_rglr_{g}"] = 0
                        else:
                            contents[f"equiv_rglr_{g}"] += rglr
                for g in range(len(info["rglr1_list"][0])):
                    wandb.summary[f"equiv_rglr_{g}"] = contents[f"equiv_rglr_{g}"]
                for lyr, rglr3 in enumerate(info["rglr3_list"]):
                    for g, rglr in enumerate(rglr3):
                        # contents[f"equiv_cos_l{lyr}g{g}"] = rglr
                        if contents.get(f"equiv_cos_{g}") is None:
                            contents[f"equiv_cos_{g}"] = 0
                        else:
                            contents[f"equiv_cos_{g}"] += rglr
                for g in range(len(info["rglr3_list"][0])):
                    wandb.summary[f"equiv_cos_{g}"] = contents[f"equiv_cos_{g}"]
                contents["l2_rglr"] = info["rglr2"]
            if ms.modelsearch_cond:
                for i, mse_by_state in enumerate(ms.valid_msebystate_list):
                    mse_by_state = np.array(mse_by_state)
                    contents[f"mse_for_state_{i-1}"] = mse_by_state
            wandb.log(contents)

        for epoch in pbar:
            if es.is_stop():
                break
            # report
            if not args.logoff:
                reporting({
                    "epoch": epoch,
                    "train_mse": np.array(train_mse),
                    "train_loss": np.array(train_loss),
                    "valid_mse": np.array(valid_mse),
                    "rglr1_list": np.array(rglr1_list),
                    "rglr2": np.array(rglr2),
                    "rglr3_list": np.array(rglr3_list),
                    "test_mse_updated": test_mse_updated,
                    "train_length": np.array(train_length),
                    "valid_length": np.array(valid_length),
                    "test_length": np.array(test_length)
                })

            # training
            train_length = 0
            train_mse = 0
            train_loss = 0
            lr = cosine_schedule(
                args.lr, epoch, num_epochs, min_value=args.min_lr)
            for train_idx, (x, y) in enumerate(trainloader):
                if args.equiv_decay:
                    equiv = cosine_schedule(equiv_coef, epoch, num_epochs)
                l, _mse, _ = train_op(jnp.array(x), jnp.array(y), lr, equiv)
                if epoch == 0 and train_idx == 0:
                    print(f"Initial Train MSE {_mse:.4f}")
                train_length += x.shape[0]
                train_mse += _mse*x.shape[0]
                train_loss += l*x.shape[0]
            train_mse /= train_length
            train_loss /= train_length

            # evaluating
            valid_length = 0
            valid_mse = 0
            ms.begin(epoch)
            for x, y in validloader:
                x, y = jnp.array(x), jnp.array(y)
                if data != "motion":
                    l = mse(x, y)
                else:
                    l = unnormalized_ade(x, y)
                valid_length += x.shape[0]
                valid_mse += l*x.shape[0]
                ms.accumulate(x, y)
            valid_mse /= valid_length
            ms.end(valid_length)
            es.update(epoch, valid_mse)

            # optimal model search for hybridsoftemlp
            optimal_state = ms.optimal_state(model.get_current_state())
            model.set_state(optimal_state)

            adjust_equiv_cond = "softemlp" in net_name and args.auto_equiv and epoch + \
                1 == args.adjust_equiv_at
            rglr_update_cond = "softemlp" in net_name and args.auto_equiv and epoch + \
                1 > args.adjust_equiv_at-100 and epoch+1 <= args.adjust_equiv_at+1
            if (not args.logoff and (not args.sweep or test_mse_updated)) or rglr_update_cond:
                rglr1_list, rglr2, rglr3_list = equiv_regularizer(
                    except_bias=args.no_bias_regular)
                ae.update_rglr(rglr1_list)

            if adjust_equiv_cond:
                equiv = ae.adjust(equiv, args.adjust_exp)

                # measure test mse
            test_mse_updated = False
            if valid_mse < top_mse and epoch+1 > 2*args.adjust_equiv_at:
                test_mse_updated = True
                top_test_epoch = epoch
                top_mse = valid_mse
                if checkpoint_path is None:
                    now = datetime.now()
                    date_time_str = now.strftime(f"%Y%m%d%H%M%S")
                    symmetry_type = normal_type
                    if dataset_name == "cossim":
                        symmetry_type = args.sym
                    elif dataset_name == "inertia":
                        symmetry_type = f"axis{args.axis}"
                    checkpoint_name = f"{symmetry_type}{dataset_name}{net_name}{date_time_str}"
                    checkpoint_path = f"checkpoints/{checkpoint_name}.npz"
                objax.io.save_var_collection(checkpoint_path, model.vars())

        objax.io.load_var_collection(checkpoint_path, model.vars())
        test_length = 0
        test_mse = 0
        for x, y in testloader:
            x, y = jnp.array(x), jnp.array(y)
            if data != "motion":
                l = mse(x, y)
            else:
                l = unnormalized_ade(x, y)
            test_length += x.shape[0]
            test_mse += l*x.shape[0]
        # test_mse /= lengths["test"]
        test_mse /= test_length
        top_test_mse = test_mse
        wandb.summary["test_mse"] = np.array(test_mse)
        wandb.summary["top_test_epoch"] = top_test_epoch

        if ood is not None:
            ood_length = 0
            ood_mse = 0
            for x, y in oodloader:
                x, y = jnp.array(x), jnp.array(y)
                l = mse(x, y)
                ood_length += x.shape[0]
                ood_mse += l*x.shape[0]
            # ood_mse /= lengths["ood"]
            ood_mse /= ood_length
            top_ood_mse = ood_mse
            wandb.summary["OOD_mse"] = np.array(ood_mse)

        wandb.finish()
        mse_list.append(top_test_mse)
        ood_list.append(top_ood_mse)
        equiv_list.append(equiv)
        print(
            f"Trial {trial+1}, Test MSE: {top_test_mse:.3e} at {top_test_epoch}")

    print("(checkpoint)")
    print("Symmetry type:", symmetry_type)
    print(f"{args.network.lower()},ch {args.ch}, bs {args.bs}, ep {args.epochs}{', earlystop' if not args.no_early_stop else ''}")
    for _mse in mse_list:
        print(_mse)
    print(f"Test MSE: {np.mean(mse_list)}±{np.std(mse_list)}")
    for _ood in ood_list:
        print(_ood)
    print(f"OOD MSE: {np.mean(ood_list)}±{np.std(ood_list)}")
    for _equiv in equiv_list:
        print(_equiv)
    f = open("./inertia_result.txt", 'a')
    f.write(watermark)
    for i in range(len(mse_list)):
        f.write(f"epoch{i}:{mse_list[i]:.3e} ")
    f.write(f'\n{watermark}')
    f.write(f"Test MSE: {np.mean(mse_list):.3e}±{np.std(mse_list):.3e}\n")
    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basic_wd",
        type=str,
        default="1",
        help="basic weight decay",  # 1 for mixedemlp, 0 for mlp
    )
    parser.add_argument(
        "--equiv_wd",
        type=str,
        default="1e-5",
        help="equiv weight decay",  # for mxiedemlp
    )
    parser.add_argument(
        "--network",
        type=str,
        default={
            "inertia": "o3subgroupsoftemlp",
            "cossim": "o3scalesubgroupsoftemlp",
            "motion": "o3subgroupsoftemlp"
        }
    )
    parser.add_argument(
        "--logoff",
        action="store_true"
    )
    parser.add_argument(
        "--equiv",  # deprecated
        type=str,
        # "200" for o3softemlp, "200,200" for o2o3softemlp, "1" for o2o3softmixedemlp, "200,0,0,0" for hybridsoftemlp
        default=""
    )
    parser.add_argument(
        "--init_equiv",
        type=float,
        default={
            "inertia": 100,
            "cossim": 0.01,
            "motion": 0.2
        }[dataset_name] if dataset_name != "motion" else {
            "symm_scale_aware": 5,
            "symm_scale_aware2": 5,
            "scale_aware": 0.3,
            "symm_aware": 0.3,
        }[normal_type]
    )
    parser.add_argument(
        "--wd",
        type=float,
        default={
            "inertia": 2e-4,
            "cossim": 2e-5,
            "motion": 0
        }[dataset_name]
    )
    parser.add_argument(
        "--gated_wd",
        type=float,
        default=1.  # 0 for any softemlp, 1 for mixedemlp
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
        default={
            "inertia": 8000,
            "se3": 10000,
            "cossim": 2000,
            "motion": 500
        }[dataset_name] if dataset_name != "motion" else {
            "symm_scale_aware": 500,
            "symm_scale_aware2": 500,
            "scale_aware": 750,
            "symm_aware": 500,
        }[normal_type]
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5
    )
    parser.add_argument(
        "--rpp_init",
        type=str,
        default={
            "inertia": "",
            "cossim": "",
            "motion": "halfsoft"
        }[dataset_name]
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0
    )
    parser.add_argument(
        "--ch",
        type=int,
        default={
            "inertia": 128,  # 89 for a half of parameters
            "cossim": 128,  # 90 for a half of parameters
            "motion": 384  # 269 for a half of parameters
        }[dataset_name]
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
        "--noise",
        type=float,
        default=0.3
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=2  # z axis
    )
    parser.add_argument(
        "--lr",
        type=float,
        default={
            "inertia": 1e-3,
            "se3": 2e-4,
            "cossim": 1e-3,
            "motion": 2e-4
        }[dataset_name]
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse"
    )
    parser.add_argument(
        "--equiv_decay",
        action="store_true"
    )
    parser.add_argument(
        "--debug",
        action="store_true"
    )
    parser.add_argument(
        "--threshold",  # maxmse minimum equivariance error
        type=str,
        default="0"
    )
    parser.add_argument(
        "--sym",
        type=str,
        default=""
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4
    )
    parser.add_argument(
        "--n_transforms",
        type=int,
        default=0
    )
    parser.add_argument(
        "--ind_shift",
        type=int,
        default=0
    )
    parser.add_argument(
        "--ood_shift",
        type=int,
        default=0
    )
    parser.add_argument(
        "--sign",
        type=int,
        default=0  # -1: negative_train/positive_test, 1: positive_train/negative_test
    )
    parser.add_argument(
        "--no_bias_regular",
        action="store_true"
    )
    parser.add_argument(
        "--bs",
        type=int,
        default={
            "inertia": 500,
            "se3": 200,
            "cossim": 200,
            "motion": 128
        }[dataset_name]
    )
    parser.add_argument(
        "--modelsearch",
        type=int,
        default=1
    )
    parser.add_argument(
        "--initial_state",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--auto_equiv",
        action="store_true"
    )
    parser.add_argument(
        "--adjust_equiv_at",
        type=int,
        default={
            "inertia": 2000,
            "se3": 2500,
            "cossim": 500,
            "motion": 100
        }[dataset_name]
    )
    parser.add_argument(
        "--adjust_exp",
        type=float,
        default={
            "inertia": 2,
            "cossim": 2,
            "motion": 5
        }[dataset_name] if dataset_name != "motion" else {
            "symm_scale_aware": 5,
            "symm_scale_aware2": 5,
            "scale_aware": 5,
            "symm_aware": 5,
        }[normal_type]
    )
    parser.add_argument(
        "--sweep",
        action="store_true"
    )
    parser.add_argument(
        "--precision",
        action="store_true"
    )
    parser.add_argument(
        "--equiv_error_test",
        action="store_true"
    )
    parser.add_argument(
        "--hypothesis_test",
        action="store_true"
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=1
    )
    parser.add_argument(
        "--error_test_samples",
        type=int,
        default=10
    )
    parser.add_argument(
        "--normal_type",  # deprecated
        type=str,
        default=""
    )
    parser.add_argument(
        "--no_early_stop",
        action="store_true"
    )
    parser.add_argument(
        "--model_equiv_error",
        action="store_true"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2022
    )
    parser.add_argument(
        "--num_frames", # only for frame averaging
        type=int,
        default=8
    )
    parser.add_argument(
        "--flex_act",
        action="store_true"
    )

    args = parser.parse_args()
    args.dataset = dataset_name
    if args.precision:
        with jax.default_matmul_precision('float32'):
            main(args)
    else:
        main(args)
