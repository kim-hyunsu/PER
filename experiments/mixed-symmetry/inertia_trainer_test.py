from lib2to3.pgen2.pgen import generate_grammar
import sys  # nopep8
sys.path.append("../trainer/")  # nopep8
sys.path.append("../")  # nopep8
sys.path.append("../../")  # nopep8
from tqdm import tqdm
import argparse
import objax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from datasets import Inertia, ModifiedInertia, SoftModifiedInertia, RandomlyModifiedInertia, NoisyModifiedInertia
from rpp.objax import (BiEMLP, HybridSoftEMLP, MixedEMLP, MixedEMLPH, MixedGroup2EMLP, MixedGroupEMLP,
                       MixedMLPEMLP, SoftEMLP, SoftMixedEMLP, SoftMultiEMLP, WeightedEMLP, MixedGroupEMLPv2, MixedGroup2EMLPv2, MultiEMLPv2)
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from utils import LoaderTo
from torch.utils.data import DataLoader
from emlp.nn import MLP, EMLP
from emlp.groups import SO2eR3, O2eR3, DkeR3, Trivial, SO, O, Embed
from rpp.groups import Union
import wandb
from functools import partial
import objax
import distrax

Oxy2 = Embed(O(2), 3, slice(2))
Oyz2 = Embed(O(2), 3, slice(1, 3))
Oxz2 = Embed(O(2), 3, slice(0, 3, 2))
# SLxy2 = Embed(SL(2), 3 , slice(2))


class RGLR():
    def __init__(self, loss_type, epsilon=0):
        self.loss_type = loss_type
        self.epsilon = epsilon

    def __call__(self, a, b):
        if self.loss_type == "mse":
            return 0.5*((a-b)**2).sum()

        elif self.loss_type == "logmse":
            logdiff = 2*jnp.log(self.epsilon+jnp.abs(a-b))
            logmse = jax.scipy.special.logsumexp(logdiff)
            return logmse

        elif self.loss_type == "csd":
            return cosine_similarity(a, b)


def cosine_similarity(a, b):
    inner = (a*b).sum()
    norm_a = jnp.linalg.norm(a)
    norm_b = jnp.linalg.norm(b)
    return 1-inner/(norm_a*norm_b)


class EquivPrior(objax.Module):
    def __init__(self, initial_equiv, **kwargs):
        self.params = kwargs
        shape = kwargs["shape"]
        scale = kwargs["scale"]
        self.prior_var = kwargs["prior_var"]
        initial_equiv = shape*scale*jnp.ones_like(initial_equiv)
        self.equiv = objax.TrainVar(initial_equiv)
        # self.dist = distrax.Gamma(
        #     shape*jnp.ones_like(initial_equiv),
        #     (1/scale)*jnp.ones_like(initial_equiv))
        self.dist = distrax.Normal(
            jnp.zeros_like(initial_equiv), scale*jnp.ones_like(initial_equiv))
        # self.dist = distrax.Normal(
        #     10*jnp.ones_like(initial_equiv), scale*jnp.ones_like(initial_equiv))

    def __call__(self, rglr_list, rglr2, **kwargs):
        rglrs = jnp.array(rglr_list)
        coeff = self.equiv.value**2+1
        nlogprob = jnp.sum(coeff*rglrs) / self.prior_var / 2
        nlogprob += args.wd*rglr2 / self.prior_var / 2
        loss1 = nlogprob
        loss2 = -self.dist.log_prob(self.equiv.value).sum()  # s prior
        nlogprob += loss2
        return loss1, loss2

    def get_value(self):
        return self.equiv.value


class VariationalDistribution(objax.Module):
    def __init__(self, model_params, equiv):
        # model_params : 11 number of DeviceArrays
        self.model_mu = objax.ModuleList(objax.TrainVar(jnp.zeros_like(p))
                                         for p in model_params)
        self.model_logsigma = objax.ModuleList(objax.TrainVar(-6*jnp.ones_like(p))
                                               for p in model_params)
        self.equiv_mu = objax.TrainVar(0.5*jnp.ones_like(equiv))
        self.equiv_logsigma = objax.TrainVar(jnp.zeros_like(equiv))
        # self.equiv_mu = objax.TrainVar(0.01*objax.random.normal(equiv.shape))
        # self.equiv_logsigma = objax.TrainVar(0.01*objax.random.normal(equiv.shape))

    def __call__(self):
        pass

    def log_prob(self, model_params, equiv):
        model_prior = (distrax.Normal(mu.value, jnp.exp(logsigma.value))
                       for mu, logsigma in zip(self.model_mu, self.model_logsigma))
        equiv_prior = distrax.Normal(
            self.equiv_mu.value, jnp.exp(self.equiv_logsigma.value))
        logprob = sum(dist.log_prob(p).sum()
                      for dist, p in zip(model_prior, model_params)) + equiv_prior.log_prob(equiv).sum()
        return logprob

    def sample(self, s):
        generator = objax.random.Generator(s)
        model_eps = [objax.random.normal(
            mu.value.shape, generator=objax.random.Generator(s+i+1))
            for i, mu in enumerate(self.model_mu)]
        equiv_eps = objax.random.normal(
            self.equiv_mu.value.shape, generator=generator)
        model_sample = (mu.value + jnp.exp(logsigma.value)*eps
                        for mu, logsigma, eps in zip(self.model_mu, self.model_logsigma, model_eps))
        equiv_sample = self.equiv_mu.value + \
            jnp.exp(self.equiv_logsigma.value)*equiv_eps
        return model_sample, equiv_sample

    def set_equiv(self, mu, logsigma):
        objax.TrainRef(self.equiv_mu).value = mu
        objax.TrainRef(self.equiv_logsigma).value = logsigma

    def set_model(self, mu, logsigma):
        for mm, m in zip(self.model_mu, mu):
            objax.TrainRef(mm).value = m
        for ml, l in zip(self.model_logsigma, logsigma):
            objax.TrainRef(ml).value = l


def main(args):

    num_epochs = args.epochs
    ndata = 1000+2000
    seed = 2022
    torch.manual_seed(seed)
    np.random.seed(seed)

    lr = args.lr

    bs = 500
    logger = []

    equiv_wd = [float(wd) for wd in args.equiv_wd.split(",")]
    basic_wd = [float(wd) for wd in args.basic_wd.split(",")]
    intervals = [int(interval) for interval in args.intervals.split(",")]

    mse_list = []
    for trial in range(args.trials):
        watermark = "var{}lr{}es{}wd{}sh{}sc{}t{}".format(
            args.likelihood_var, args.lr, args.ensemble, args.wd, args.param_shape, args.param_scale, trial)

        wandb.init(
            project="Mixed Symmetry, Inertia, Bayesian",
            name=watermark,
            mode="disabled"
        )
        wandb.config.update(args)

        # Initialize dataset with 1000 examples
        if args.soft == True:
            print('soft inertia is selected')
            dset = SoftModifiedInertia(3000, noise=args.noise)
        elif args.noise_std == 0:
            dset = ModifiedInertia(3000, noise=args.noise, axis=args.axis)
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
            args.basic_wd = [0.]
            model = MLP(dset.rep_in, dset.rep_out,
                        group=G, num_layers=3, ch=args.ch)
        elif args.network.lower() == "emlp":
            model = EMLP(dset.rep_in, dset.rep_out,
                         group=G, num_layers=3, ch=args.ch)
        # mixedemlp == rpp
        elif args.network.lower() == 'mixedemlp':
            model = MixedEMLP(dset.rep_in, dset.rep_out,
                              group=G, num_layers=3, ch=args.ch,
                              gnl=args.gatednonlinearity)
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
        elif args.network.lower() == "oxy2oyz2oxz2softemlp":
            G = (Oxy2, Oyz2, Oxz2)
            model = SoftMultiEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init)
        elif args.network.lower() == "oxy2o3softmixedemlp":
            G = (Oxy2, O(3))
            model = SoftMixedEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init)
        elif args.network.lower() == "oyz2o3softmixedemlp":
            G = (Oyz2, O(3))
            model = SoftMixedEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init)
        elif args.network.lower() == "oxz2o3softmixedemlp":
            G = (Oxz2, O(3))
            model = SoftMixedEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init)
        elif args.network.lower() == "exp:partialprojection":
            G = (O(3), Oxy2, Oyz2, Oxz2)
            model = SoftMixedEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=True,
                                  rpp_init=args.rpp_init)
        elif args.network.lower() == "exp:partialprojection2":
            G = (O(3), Oxy2, Oyz2, Oxz2)
            model = HybridSoftEMLP(dset.rep_in, dset.rep_out,
                                   groups=G, num_layers=3, ch=args.ch,
                                   gnl=args.gatednonlinearity,
                                   rpp_init=args.rpp_init)
        elif args.network.lower() == "hybridsoftemlp":
            G = (O(3), Oxy2, Oyz2, Oxz2)
            model = HybridSoftEMLP(dset.rep_in, dset.rep_out,
                                   groups=G, num_layers=3, ch=args.ch,
                                   gnl=args.gatednonlinearity,
                                   rpp_init=args.rpp_init)
        elif args.network.lower() == "bimixedemlp":
            G = (O(3), Oxy2)
            model = BiEMLP(dset.rep_in, dset.rep_out,
                           groups=G, num_layers=3, ch=args.ch,
                           gnl=args.gatednonlinearity)
        elif args.network.lower() == "o3subgroupsoftemlp":
            G = (O(3), Oxy2, Oyz2, Oxz2)
            model = SoftMultiEMLP(dset.rep_in, dset.rep_out,
                                  groups=G, num_layers=3, ch=args.ch,
                                  gnl=args.gatednonlinearity,
                                  rpp_init=args.rpp_init)
        else:
            raise Exception()

        equivlength = len(G) if isinstance(G, tuple) else 1
        equiv_coef = [0. for _ in range(equivlength)]
        for i, eq in enumerate(args.equiv.split(",")):
            equiv_coef[i] = float(eq)
        equiv_coef = jnp.array(equiv_coef)
        equiv = EquivPrior(
            equiv_coef,
            shape=args.param_shape, scale=args.param_scale,
            prior_var=args.likelihood_var/len(trainloader.dataset))

        guide = VariationalDistribution(
            model.vars().tensors(), equiv.get_value())

        if args.aug:
            assert not isinstance(G, tuple)
            model = dset.default_aug(model)

        rglr_func = RGLR(args.loss_type, epsilon=1e-45)

        @ objax.Jit
        @ objax.Function.with_vars(guide.vars())
        def equiv_regularizer():
            net_name = args.network.lower()
            rglr1_list = [0 for _ in range(equivlength)]
            rglr2 = 0
            rglr3_list = [0 for _ in range(equivlength)]
            if net_name in ["exp:partialprojection2"]:
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W = lyr.w.value.reshape(-1)
                    b = lyr.b.value
                    Pw = lyr.Pw_list[0]
                    Pb = lyr.Pb_list[0]
                    W1 = Pw@W
                    b1 = Pb@b
                    rglr1_list[0] += 0.5 * \
                        ((W-W1)**2).sum() + 0.5*((b-b1)**2).sum()
                    rglr2 += 0.5*(W**2).sum() + 0.5*(b**2).sum()
            elif "softemlp" in net_name or net_name in []:
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W = lyr.w.value.reshape(-1)
                    b = lyr.b.value
                    Pw_list = lyr.Pw_list
                    Pb_list = lyr.Pb_list
                    for j, (Pw, Pb) in enumerate(zip(Pw_list, Pb_list)):
                        W1 = Pw@W
                        b1 = Pb@b
                        # rglr1_list[i] += 0.5 * \
                        #     ((W-W1)**2).sum() + 0.5*((b-b1)**2).sum()
                        rglr1_list[j] += rglr_func(W, W1) + rglr_func(b, b1)
                        # rglr3_list[j] += cosine_similarity(W, W1)
                    rglr2 += 0.5*(W**2).sum() + 0.5*(b**2).sum()
            elif "softmixedemlp" in net_name or net_name in []:
                for i, lyr in enumerate(model.network):
                    if i != len(model.network)-1:
                        lyr = lyr.linear
                    W = lyr.w.value.reshape(-1)
                    b = lyr.b.value
                    Pw_list = lyr.Pw_list
                    Pb_list = lyr.Pb_list
                    for j, (Pw1, Pb1) in enumerate(zip(Pw_list, Pb_list)):
                        W1 = Pw1@lyr.Pw@W
                        Wdiff = W-W1
                        b1 = Pb1@lyr.Pb@b
                        bdiff = b-b1
                        rglr1_list[j] += 0.5*(Wdiff*(lyr.Pw@Wdiff)).sum() + \
                            0.5*(bdiff*(lyr.Pb@bdiff)).sum()
                    rglr2 += 0.5*(W**2).sum() + 0.5*(b**2).sum()

            return rglr1_list, rglr2, rglr3_list

        equiv_samples_list = [
            jnp.array([0., 0., 0.]),
            jnp.array([0.7, 0.7, 0.7]),
            jnp.array([9.95, 0.7, 0.7]),
            jnp.array([0.7, 9.95, 0.7]),
            jnp.array([0.7, 0.7, 9.95]),
            jnp.array([9.95, 9.95, 9.95]),
        ]

        objax.io.load_var_collection(args.checkpoint, model.vars())
        model_tensor = model.vars().tensors()

        guide.set_model(model_tensor, [jnp.log(
            args.w_std)*jnp.ones_like(p) for p in model_tensor])

        @ objax.Jit
        @ objax.Function.with_vars(guide.vars())
        def loss(x, y):
            elbo_list = []
            for i, equiv_samples in enumerate(equiv_samples_list):
                for var in equiv.vars().subset(objax.TrainVar):
                    objax.TrainRef(var).value = equiv_samples
                guide.set_equiv(equiv_samples, jnp.log(
                    args.s_std)*jnp.ones_like(equiv_samples))

                yhat = model(x)
                mse = ((yhat-y)**2).mean()  # -logP(D|W)
                loss1 = mse

                rglr = 0
                rglr1_list, rglr2, _ = equiv_regularizer()
                # -logP(W|s)
                loss2, loss3 = equiv(rglr1_list, rglr2)
                rglr += loss2 + loss3
                loss4 = guide.log_prob(  # logQ(s,W)
                    model.vars().tensors(), equiv.get_value())
                rglr += loss4
                l_data = len(trainloader.dataset)
                var = args.likelihood_var
                _loss = ((0.5/var)*mse + (1/l_data)*rglr)
                loss1 *= (0.5/var)
                loss2 *= (1/l_data)
                loss3 *= (1/l_data)
                loss4 *= (1/l_data)
                loss_list = jnp.array([_loss, loss1, loss2, loss3, loss4])
                elbo_list.append(loss_list)

            return jnp.array(elbo_list)

        statelength = len(G)+1 if isinstance(G, tuple) else 2
        train_mse = 0
        for x, y in trainloader:
            l = loss(jnp.array(x), jnp.array(y))
            train_mse += l*x.shape[0]
        train_mse /= len(trainloader.dataset)
        for s, elbo in zip(equiv_samples_list, train_mse):
            print("{}: ELBO {:.5f} p(D|s,w) {:.5f} p(w|s) {:.5f} p(s) {:.5f} q(s,w) {:.5f}".format(
                s, elbo[0], elbo[1], elbo[2], elbo[3], elbo[4]
            ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="modified inertia ablation")
    parser.add_argument(
        "--experiment",
        type=str,
        default="unbal-",
        help="type of network {per-, unbal-, mis-}",
    )
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
        default="200"  # "200" for o3softemlp, "200,200" for o2o3softemlp, "1" for o2o3softmixedemlp, "200,0,0,0" for hybridsoftemlp
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0  # 0 or 1e-5
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
        default=8000
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
        "--soft",
        action="store_true"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse"
    )
    parser.add_argument(
        "--param_shape",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--param_scale",
        type=float,
        default=10
    )
    parser.add_argument(
        "--ensemble",
        type=int,
        default=2
    )
    parser.add_argument(
        "--likelihood_var",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
    )
    parser.add_argument(
        "--s_std",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--w_std",
        type=float,
        default=0.05
    )
    args = parser.parse_args()

    main(args)
