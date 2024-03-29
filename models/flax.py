
from builtins import isinstance
import jax
import jax.numpy as jnp
import numpy as np
from emlp.reps import T, Rep, Scalar
from emlp.reps import bilinear_weights
# from emlp.reps import LinearOperator # why does this not work?
from emlp.reps.linear_operator_base import LinearOperator
from emlp.reps.product_sum_reps import SumRep
from emlp.groups import Group
from oil.utils.utils import Named, export
from flax import linen as nn
import logging
from emlp.nn import gated, gate_indices, uniform_rep
from models.objax import reset_solcache, uniform_reps
from typing import Union, Iterable, Optional, List
# def Sequential(*args):
#     """ Wrapped to mimic pytorch syntax"""
#     return nn.Sequential(args)


RPP_SCALE = 1e-3


@export
def Linear(repin, repout):
    """ Basic equivariant Linear layer from repin to repout."""
    cin = repin.size()
    cout = repout.size()
    rep_W = repin >> repout
    Pw = rep_W.equivariant_projector()
    Pb = repout.equivariant_projector()
    logging.info(f"Linear W components:{rep_W.size()} rep:{rep_W}")
    return _Linear(Pw, Pb, cout)


class _Linear(nn.Module):
    Pw: LinearOperator
    Pb: LinearOperator
    cout: int

    @nn.compact
    def __call__(self, x):
        w = self.param('w', nn.initializers.lecun_normal(),
                       (self.cout, x.shape[-1]))
        b = self.param('b', nn.initializers.zeros, (self.cout,))
        W = (self.Pw@w.reshape(-1)).reshape(*w.shape)
        B = self.Pb@b
        return x@W.T+B


@export
def BiLinear(repin, repout):
    """ Cheap bilinear layer (adds parameters for each part of the input which can be
        interpreted as a linear map from a part of the input to the output representation)."""
    Wdim, weight_proj = bilinear_weights(repout, repin)
    # self.w = TrainVar(objax.random.normal((Wdim,)))#xavier_normal((Wdim,))) #TODO: revert to xavier
    logging.info(f"BiW components: dim:{Wdim}")
    return _BiLinear(Wdim, weight_proj)


class _BiLinear(nn.Module):
    Wdim: int
    weight_proj: callable

    @nn.compact
    def __call__(self, x):
        # TODO: change to standard normal
        w = self.param('w', nn.initializers.normal(1.0), (self.Wdim,))
        W = self.weight_proj(w, x)
        out = .03*(W@x[..., None])[..., 0]
        return out


@export
# TODO: add support for mixed tensors and non sumreps
class GatedNonlinearity(nn.Module):
    """ Gated nonlinearity. Requires input to have the additional gate scalars
        for every non regular and non scalar rep. Applies swish to regular and
        scalar reps. (Right now assumes rep is a SumRep)"""
    rep: Rep

    def __call__(self, values):
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = jax.nn.sigmoid(
            gate_scalars) * values[..., :self.rep.size()]
        return activations


@export
class RPPGatedNonlinearity(nn.Module):
    rep: Rep

    @nn.compact
    def __call__(self, values):
        ch = self.rep.size()
        basic_init = lambda *args, **kwargs: nn.initializers.ones(
            *args, **kwargs)*RPP_SCALE
        w = self.param('w_basic', basic_init, (ch,))
        gate_scalars = values[..., gate_indices(self.rep)]
        gated_activations = jax.nn.sigmoid(
            gate_scalars) * values[..., :self.rep.size()]
        return gated_activations+w*swish(values[..., :self.rep.size()])


@export
def EMLPBlock(rep_in, rep_out):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """
    linear = Linear(rep_in, gated(rep_out))
    bilinear = BiLinear(gated(rep_out), gated(rep_out))
    nonlinearity = GatedNonlinearity(rep_out)
    return _EMLPBlock(linear, bilinear, nonlinearity)


class _EMLPBlock(nn.Module):
    linear: nn.Module
    bilinear: nn.Module
    nonlinearity: nn.Module

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


@export
def EMLP(rep_in, rep_out, group, ch=384, num_layers=3):
    """ Equivariant MultiLayer Perceptron. 
        If the input ch argument is an int, uses the hands off uniform_rep heuristic.
        If the ch argument is a representation, uses this representation for the hidden layers.
        Individual layer representations can be set explicitly by using a list of ints or a list of
        representations, rather than use the same for each hidden layer.
        Args:
            rep_in (Rep): input representation
            rep_out (Rep): output representation
            group (Group): symmetry group
            ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
            num_layers (int): number of hidden layers
        Returns:
            Module: the EMLP objax module."""
    logging.info("Initing EMLP (flax)")
    rep_in = rep_in(group)
    rep_out = rep_out(group)
    if isinstance(ch, int):
        middle_layers = num_layers*[uniform_rep(ch, group)]
    elif isinstance(ch, Rep):
        middle_layers = num_layers*[ch(group)]
    else:
        middle_layers = [(c(group) if isinstance(c, Rep)
                          else uniform_rep(c, group)) for c in ch]
    reps = [rep_in]+middle_layers
    logging.info(f"Reps: {reps}")
    return Sequential(*[EMLPBlock(rin, rout) for rin, rout in zip(reps, reps[1:])], Linear(reps[-1], rep_out))


@export
def MixedLinear(repin, repout):
    """ Basic equivariant Linear layer from repin to repout."""
    cin = repin.size()
    cout = repout.size()
    rep_W = repin >> repout
    Pw = rep_W.equivariant_projector()
    Pb = repout.equivariant_projector()
    logging.info(f"Linear W components:{rep_W.size()} rep:{rep_W}")
    return _MixedLinear(Pw, Pb, cout)


class _MixedLinear(nn.Module):
    Pw: LinearOperator
    Pb: LinearOperator
    cout: int

    @nn.compact
    def __call__(self, x):
        basic_init = lambda *args, **kwargs: nn.initializers.lecun_normal()(*args, **
                                                                            kwargs)*RPP_SCALE
        w_equiv = self.param(
            'w_equiv', nn.initializers.lecun_normal(), (x.shape[-1], self.cout))
        w_basic = self.param('w_basic', basic_init, (x.shape[-1], self.cout))
        b_equiv = self.param('b_equiv', nn.initializers.zeros, (self.cout,))
#         b_basic = self.param('b_basic',nn.initializers.zeros,(self.cout,))
        b_basic = self.param('b_basic', basic_init, (self.cout, 1))
        W = (self.Pw@w_equiv.reshape(-1)).reshape(*w_equiv.shape)
        B = self.Pb@b_equiv
        return x@(W + w_basic) + B + b_basic[:, 0]


@export
def SoftEMLPLinear(repin_list, repout_list):
    cin = repin_list[0].size()
    cout = repout_list[0].size()
    rep_W_list = [repin >> repout
                  for repin, repout in zip(repin_list, repout_list)]
    rep_b_list = repout_list
    Pw_list = []
    Pb_list = []
    for rep_W, rep_b in zip(rep_W_list, rep_b_list):
        rep_W = reset_solcache(rep_W)
        rep_b = reset_solcache(rep_b)
        Pw_list.append(rep_W.equivariant_projector())
        Pb_list.append(rep_b.equivariant_projector())

    init_state = -jnp.ones((), dtype=np.int16)
    return _SoftEMLPLinear(Pw_list, Pb_list, cout, init_state)


class _SoftEMLPLinear(nn.Module):
    Pw_list: List[LinearOperator]
    Pb_list: List[LinearOperator]
    cout: int
    state: jax._src.device_array.DeviceArray

    @nn.compact
    def __call__(self, x):
        w = self.param('w', nn.initializers.lecun_normal(),
                       (x.shape[-1], self.cout))
        b = self.param('b', nn.initializers.zeros, (self.cout,))
        # if self.state <0:
        #     return x@w + b
        # else:
        #     Pw = self.Pw_list[self.state]
        #     Pb = self.Pb_list[self.state]
        #     W = (Pw@w.reshape(-1)).reshape(*w.shape)
        #     B = Pb@b
        #     return x@W + B
        Pw = self.Pw_list[self.state]
        Pb = self.Pb_list[self.state]
        W = (Pw@w.reshape(-1)).reshape(*w.shape)
        B = Pb@b
        mlp = self.state < 0
        W = mlp*w + (1-mlp)*W
        B = mlp*b + (1-mlp)*B
        return x@W+B

    def get_params(self):
        W = self.variables['params']['w']
        b = self.variables['params']['b']
        return W,b

    def __hash__(self):
        return id(self)

    def set_state(self, state):
        self.state = state

    def get_current_state(self):
        return self.state


def MixedEMLPBlock(rep_in, rep_out):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """
    mixedlinear = MixedLinear(rep_in, gated(rep_out))
    bilinear = BiLinear(gated(rep_out), gated(rep_out))
    nonlinearity = RPPGatedNonlinearity(rep_out)
    return _MixedEMLPBlock(mixedlinear, bilinear, nonlinearity)


class _MixedEMLPBlock(nn.Module):
    mixedlinear: nn.Module
    bilinear: nn.Module
    nonlinearity: nn.Module

    def __call__(self, x):
        lin = self.mixedlinear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


def SoftEMLPBlock(rep_in_list, rep_out_list, gnl):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """
    mixedlinear = SoftEMLPLinear(
        rep_in_list, [gated(rep_out) for rep_out in rep_out_list])
    bilinear = BiLinear(gated(rep_out_list[0]), gated(rep_out_list[0]))
    nonlinearity = RPPGatedNonlinearity(
        rep_out_list[0]) if not gnl else GatedNonlinearity(rep_out_list[0])
    return _SoftEMLPBlock(mixedlinear, bilinear, nonlinearity)


class _SoftEMLPBlock(nn.Module):
    linear: nn.Module
    bilinear: nn.Module
    nonlinearity: nn.Module

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)

    def set_state(self, state):
        self.linear.set_state(state)
    def get_current_state(self):
        return self.linear.get_current_state()


@export
def MixedEMLP(rep_in, rep_out, group, ch=384, num_layers=3):
    """ Equivariant MultiLayer Perceptron. 
        If the input ch argument is an int, uses the hands off uniform_rep heuristic.
        If the ch argument is a representation, uses this representation for the hidden layers.
        Individual layer representations can be set explicitly by using a list of ints or a list of
        representations, rather than use the same for each hidden layer.
        Args:
            rep_in (Rep): input representation
            rep_out (Rep): output representation
            group (Group): symmetry group
            ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
            num_layers (int): number of hidden layers
        Returns:
            Module: the EMLP objax module."""
    logging.info("Initing EMLP (flax)")
    rep_in = rep_in(group)
    rep_out = rep_out(group)
    if isinstance(ch, int):
        middle_layers = num_layers*[uniform_rep(ch, group)]
    elif isinstance(ch, Rep):
        middle_layers = num_layers*[ch(group)]
    else:
        middle_layers = [(c(group) if isinstance(c, Rep)
                          else uniform_rep(c, group)) for c in ch]
    reps = [rep_in]+middle_layers
    logging.info(f"Reps: {reps}")
    return Sequential(*[MixedEMLPBlock(rin, rout) for rin, rout in zip(reps, reps[1:])], MixedLinear(reps[-1], rep_out))


@export
def SoftEMLP(rep_in, rep_out, groups, ch=384, num_layers=3, gnl=False):
    rep_in_list = [rep_in(g) for g in groups]
    rep_out_list = [rep_out(g) for g in groups]
    if isinstance(ch, int):
        middle_layers_list = [num_layers*[sum_rep]
                              for sum_rep in uniform_reps(ch, groups)]
    elif isinstance(ch, Rep):
        middle_layers_list = [num_layers*[ch(g)] for g in groups]
    else:
        middle_layers_list = [[] for _ in range(len(groups))]
        for c in ch:
            if isinstance(c, Rep):
                for i, g in enumerate(groups):
                    middle_layers_list[i].append(c(g))
            else:
                for i, sum_rep in enumerate(uniform_reps(c, groups)):
                    middle_layers_list[i].append(num_layers*[sum_rep])

    reps_list = [[rep_in]+middle_layers
                 for rep_in, middle_layers in zip(rep_in_list, middle_layers_list)]
    rin_list = []
    rout_list = []
    for i in range(len(reps_list[0])-1):
        rins = []
        routs = []
        for j in range(len(groups)):
            rins.append(reps_list[j][i])
            routs.append(reps_list[j][i+1])
        rin_list.append(rins)
        rout_list.append(routs)
    return Sequential(
        *[SoftEMLPBlock(rins, routs, gnl)
          for rins, routs in zip(rin_list, rout_list)], SoftEMLPLinear(rout_list[-1], rep_out_list))


def swish(x):
    return jax.nn.sigmoid(x)*x


class _Sequential(nn.Module):
    modules: Iterable[callable]

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def set_state(self, state):
        for module in self.modules:
            module.set_state(state)

    def get_current_state(self):
        return self.modules[0].get_current_state()


def Sequential(*layers):
    return _Sequential(layers)


def MLPBlock(cout):
    # ,nn.BatchNorm0D(cout,momentum=.9),swish)#,
    return Sequential(nn.Dense(cout), swish)


@export
class MLP(nn.Module, metaclass=Named):
    """ Standard baseline MLP. Representations and group are used for shapes only. """
    rep_in: Rep
    rep_out: Rep
    group: Group
    ch: Optional[InterruptedError] = 384
    num_layers: Optional[int] = 3

    def setup(self):
        logging.info("Initing MLP (flax)")
        cout = self.rep_out(self.group).size()
        hidden_units = self.num_layers * \
            [self.ch] if isinstance(self.ch, int) else self.ch
        self.modules = [MLPBlock(ch) for ch in hidden_units]+[nn.Dense(cout)]

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x
