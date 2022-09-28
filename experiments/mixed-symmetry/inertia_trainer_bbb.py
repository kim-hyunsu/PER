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
        loc = kwargs["loc"]
        self.prior_var = kwargs["prior_var"]
        initial_equiv = shape*scale*jnp.ones_like(initial_equiv)
        self.equiv = objax.TrainVar(initial_equiv)
        # self.dist = distrax.Gamma(
        #     shape*jnp.ones_like(initial_equiv),
        #     (1/scale)*jnp.ones_like(initial_equiv))
        self.dist = distrax.Normal(
            loc*jnp.ones_like(initial_equiv), scale*jnp.ones_like(initial_equiv))

    def __call__(self, rglr_list, rglr2, **kwargs):
        rglrs = jnp.array(rglr_list)
        coeff = self.equiv.value**2+1
        nlogprob = jnp.sum(coeff*rglrs) / self.prior_var / 2
        nlogprob += args.wd*rglr2 / self.prior_var / 2
        nlogprob -= self.dist.log_prob(self.equiv.value).sum()  # s prior
        return nlogprob

    def get_value(self):
        return self.equiv.value


class VariationalDistribution(objax.Module):
    def __init__(self, model_params, equiv, **kwargs):
        equiv_mean = kwargs["equiv_mean"]
        # model_params : 11 number of DeviceArrays
        self.model_mu = objax.ModuleList(objax.TrainVar(jnp.zeros_like(p))
                                         for p in model_params)
        self.model_logsigma = objax.ModuleList(objax.TrainVar(-6*jnp.ones_like(p))
                                               for p in model_params)
        self.equiv_mu = objax.TrainVar(equiv_mean*jnp.ones_like(equiv))
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
        ########## TODO TEMPORARY ###########
        equiv_eps = jnp.zeros_like(equiv_eps)
        #####################################
        model_sample = (mu.value + jnp.exp(logsigma.value)*eps
                        for mu, logsigma, eps in zip(self.model_mu, self.model_logsigma, model_eps))
        equiv_sample = self.equiv_mu.value + \
            jnp.exp(self.equiv_logsigma.value)*equiv_eps
        ########## TODO TEMPORARY ###########
        equiv_sample = jax.lax.stop_gradient(equiv_sample)
        #####################################
        return model_sample, equiv_sample


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
        watermark = "var{}lr{}es{}wd{}sh{}sc{}lc{}t{}".format(
            args.likelihood_var, args.lr, args.ensemble, args.wd, args.param_shape, args.param_scale, args.param_loc, trial)

        wandb.init(
            project="Mixed Symmetry, Inertia, Bayesian",
            name=watermark,
            mode="disabled" if args.logoff else "online"
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
        if args.equiv == "":
            equiv_coef = args.param_loc*jnp.ones((equivlength,))
        else:
            for i, eq in enumerate(args.equiv.split(",")):
                equiv_coef[i] = float(eq)
            equiv_coef = jnp.array(equiv_coef)
        equiv = EquivPrior(
            equiv_coef,
            shape=args.param_shape, scale=args.param_scale, loc=args.param_loc,
            prior_var=args.likelihood_var/len(trainloader.dataset))

        guide = VariationalDistribution(
            model.vars().tensors(), equiv.get_value(),
            equiv_mean=equiv_coef)

        # foo = model.vars().tensors()
        # #foo: tensors()
        # print("foo", foo[1])
        # model_samples, _ = guide.sample()
        # #model_samples: tensors()
        # train_vars = (objax.TrainRef(x) for x in model.vars().subset(objax.TrainVar))
        # for v, s in zip(train_vars, model_samples):
        #     v.value = s
        # bar = model.vars().tensors()
        # print("bar", bar[1])

        if args.aug:
            assert not isinstance(G, tuple)
            model = dset.default_aug(model)

        opt = objax.optimizer.Adam(guide.vars())  # ,beta2=.99)

        def cosine_schedule(init_value, current_steps, total_steps, alpha=0.0):
            cosine_decay = 0.5 * \
                (1 + jnp.cos(jnp.pi * current_steps / total_steps))
            decayed = (1 - alpha) * cosine_decay + alpha
            return init_value * decayed

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

        @ objax.Jit
        @ objax.Function.with_vars(guide.vars())
        def msebystate(x, y):
            net_name = args.network.lower()
            if net_name in ["hybridsoftemlp"]:
                current_state = model.get_current_state()
                msebystate_list = []
                for state in range(-1, 4):
                    model.set_state(state)
                    yhat_prime = model(x)
                    msebystate_list.append(((yhat_prime-y)**2).mean())
                model.set_state(current_state)
            elif net_name in ["exp:partialprojection"]:
                msebystate_list = []
                for idx in range(0, 4):
                    if idx != 0:
                        proj = model.get_proj(idx)
                        model.set_proj(proj)
                    yhat_prime = model(x)
                    msebystate_list.append(((yhat_prime-y)**2).mean())
                model.reset_proj()
            elif net_name in ["exp:partialprojection2"]:
                msebystate_list = []
                for idx in range(0, 4):
                    model.set_state(idx)
                    yhat_prime = model(x)
                    msebystate_list.append(((yhat_prime-y)**2).mean())
                model.set_state(-1)
            else:
                msebystate_list = [0. for _ in range(equivlength)]

            return msebystate_list

        @ objax.Jit
        @ objax.Function.with_vars(guide.vars())
        def mse(x, y, epoch):
            yhat = 0
            for k in range(args.ensemble):
                model_samples, equiv_samples = guide.sample(epoch*(k+1))
                for var, s in zip(model.vars().subset(objax.TrainVar), model_samples):
                    objax.TrainRef(var).value = s
                for var in equiv.vars().subset(objax.TrainVar):
                    objax.TrainRef(var).value = equiv_samples
                yhat += model(x) / args.ensemble
            return ((yhat-y)**2).mean()

        @ objax.Jit
        @ objax.Function.with_vars(guide.vars())
        def loss(x, y, epoch):
            _loss = 0
            for k in range(args.ensemble):
                model_samples, equiv_samples = guide.sample(epoch*(k+1))
                for var, s in zip(model.vars().subset(objax.TrainVar), model_samples):
                    objax.TrainRef(var).value = s
                for var in equiv.vars().subset(objax.TrainVar):
                    objax.TrainRef(var).value = equiv_samples
                yhat = model(x)
                mse = ((yhat-y)**2).mean()  # -logP(D|W)

                rglr = 0
                net_name = args.network.lower()
                if "soft" in net_name or net_name in ["exp:partialprojection2"]:
                    rglr1_list, rglr2, _ = equiv_regularizer()
                    # -logP(W|s)
                    rglr += equiv(rglr1_list, rglr2)
                    rglr += guide.log_prob(  # logQ(s,W)
                        model.vars().tensors(), equiv.get_value())
                elif "mixedemlp" in net_name or net_name in []:
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

                l_data = len(trainloader.dataset)
                var = args.likelihood_var
                _loss += ((0.5/var)*mse + (1/l_data)*rglr)/args.ensemble

            return _loss

        grad_and_val = objax.GradValues(loss, guide.vars())

        @ objax.Jit
        @ objax.Function.with_vars(guide.vars()+opt.vars())
        def train_op(x, y, lr, epoch):
            g, v = grad_and_val(x, y, epoch)
            opt(lr=lr, grads=g)
            return v

        top_mse = float('inf')
        top_test_mse = float('inf')
        top_test_epoch = -1
        pbar = tqdm(range(num_epochs))
        statelength = len(G)+1 if isinstance(G, tuple) else 2
        top_msebystate_list = [float('inf') for _ in range(statelength)]
        for epoch in pbar:

            # begin = [0]+intervals[:-1]
            # end = intervals
            # interval_idx = len(equiv_coef)
            # for i, (b, e) in enumerate(zip(begin, end)):
            #     if b <= epoch < e+1:
            #         interval_idx = i

            # equiv = jnp.zeros_like(equiv_coef)
            # equiv = equiv.at[:interval_idx].set(equiv_coef[:interval_idx])

            # different regularization for each interval
            # begin = [0]+intervals[:-1]
            # end = intervals
            # interval_idx = 0
            # for i, (b, e) in enumerate(zip(begin, end)):
            #     if b <= epoch < e+1:
            #         interval_idx = len(equiv_coef)-i
            # equiv = jnp.zeros_like(equiv_coef)
            # equiv = equiv.at[interval_idx:].set(equiv_coef[interval_idx:])

            contents = dict()
            contents["state"] = model.get_current_state() if hasattr(
                model, 'get_current_state') else -2

            # training
            train_mse = 0
            lr = cosine_schedule(args.lr, epoch, num_epochs)
            contents["lr"] = lr
            for x, y in trainloader:
                l = train_op(jnp.array(x), jnp.array(y), lr, epoch)
                train_mse += l[0]*x.shape[0]
            train_mse /= len(trainloader.dataset)

            # evaluating
            net_name = args.network.lower()
            modelsearch_cond = (
                epoch+1) % 10 == 0 and net_name in ["hybridsoftemlp"]
            valid_mse = 0
            valid_msebystate_list = [0 for _ in range(statelength)]
            for x, y in validloader:
                x, y = jnp.array(x),  jnp.array(y)
                l = mse(x, y, epoch)
                valid_mse += l*x.shape[0]
                if modelsearch_cond or net_name in ["exp:partialprojection", "exp:partialprojection2"]:
                    msebystate_list = msebystate(x, y)
                    for i, mse_by_state in enumerate(msebystate_list):
                        valid_msebystate_list[i] += mse_by_state*x.shape[0]
            valid_mse /= len(validloader.dataset)
            if modelsearch_cond or net_name in ["exp:partialprojection", "exp:partialprojection2"]:
                for i in range(statelength):
                    valid_msebystate_list[i] /= len(validloader.dataset)
                    if top_msebystate_list[i] > valid_msebystate_list[i]:
                        top_msebystate_list[i] = valid_msebystate_list[i]
            # if not args.logoff:
            #     rglr1_list, rglr2, _ = equiv_regularizer()

            # optimal model search for hybridsoftemlp
            if modelsearch_cond:
                ############### version 1 ###############
                # optimal_state = min(range(statelength),
                #                     key=lambda i: top_msebystate_list[i])-1
                # reset_cond = optimal_state == -1
                # transition_cond = model.get_current_state() == -1
                # if reset_cond or transition_cond:
                #     model.set_state(optimal_state)

                ############### version 2 ################
                optimal_state = min(range(statelength),
                                    key=lambda i: top_msebystate_list[i])-1
                model.set_state(optimal_state)

            # report
            if not args.logoff:
                contents["train_mse"] = train_mse
                contents["valid_mse"] = valid_mse
                # for i, rglr1 in enumerate(rglr1_list):
                #     contents[f"equiv_rglr_{i}"] = rglr1
                for i, eq in enumerate(guide.equiv_mu.value):
                    contents[f"equiv_coeff_{i}"] = eq
                for i, logsigma in enumerate(guide.equiv_logsigma.value):
                    contents[f"equiv_coeffstd_{i}"] = jnp.exp(logsigma)
                # contents["l2_rglr"] = rglr2
                # for i, mse_by_state in enumerate(valid_msebystate_list):
                #     contents[f"mse_for_state_{i-1}"] = mse_by_state
                wandb.log(contents)

            # measure test mse
            if valid_mse < top_mse:
                top_mse = valid_mse
                test_mse = 0
                for x, y in testloader:
                    l = mse(jnp.array(x), jnp.array(y), epoch)
                    test_mse += l*x.shape[0]
                test_mse /= len(testloader.dataset)
                top_test_mse = test_mse
                top_test_epoch = epoch
                wandb.config.update(
                    {"top_test_mse": top_test_mse}, allow_val_change=True)

        wandb.finish()
        mse_list.append(top_test_mse)
        print(
            f"Trial {trial+1}, Test MSE: {top_test_mse:.3e} at {top_test_epoch}")

    # print(mse_list)
    print(f"Test MSE: {np.mean(mse_list)}±{np.std(mse_list)}")
    f = open("./inertia_result.txt", 'a')
    f.write(watermark)
    for i in range(len(mse_list)):
        f.write(f"epoch{i}:{mse_list[i]:.3e} ")
    f.write(f'\n{watermark}')
    f.write(f"Test MSE: {np.mean(mse_list):.3e}±{np.std(mse_list):.3e}\n")
    f.close()


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
        default=""
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
        "--param_loc",
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
    args = parser.parse_args()

    main(args)
