from scipy.special import binom
from emlp.reps import T
from emlp.reps.linear_operators import lazify
import numpy as np
from numpy import isin
from objax.variable import TrainVar
from objax.nn.init import orthogonal
import objax.nn.layers as objax_layers
from objax.module import Module
import objax
import jax.numpy as jnp
import emlp.nn.objax as nn
from emlp.reps.representation import Rep, ScalarRep, Base
from emlp.groups import noise2samples
from oil.utils.utils import Named, export
import logging
import jax
from models.groups import SE3
import types

RPP_SCALE = 1e-3
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


def extend_dim_mask(repin):
    mask = jnp.ones(repin.size())
    value = jnp.zeros(repin.size())
    begin = 0
    for rep, count in repin.reps.items():
        d = rep.G.d
        for i in range(count):
            size = rep.size()
            end = begin+size
            if size == 1:
                begin = end
            elif size == d:
                mask = mask.at[end-1].set(0)
                value = value.at[end-1].set(1)
                begin = end
            elif size == d*d:
                temp = jnp.ones((d, d))
                temp = temp.at[:, -1:].set(0)
                temp = temp.at[-1:, :].set(0).reshape(-1)
                temp1 = jnp.zeros((d, d))
                temp1 = temp1.at[-1, -1].set(1).reshape(-1)
                mask = mask.at[begin:end].set(temp)
                value = value.at[begin:end].set(temp1)
                begin = end
            else:
                begin = end
                raise Exception()
    return mask, value


def rng_samples(self, N, seed):
    rng = np.random.default_rng(seed)
    """ Draw N samples from the group (not necessarily Haar measure)"""
    A_dense = jnp.stack([Ai@jnp.eye(self.d) for Ai in self.lie_algebra]
                        ) if len(self.lie_algebra) else jnp.zeros((0, self.d, self.d))
    h_dense = jnp.stack([hi@jnp.eye(self.d) for hi in self.discrete_generators]
                        ) if len(self.discrete_generators) else jnp.zeros((0, self.d, self.d))
    z = rng.standard_normal((N, A_dense.shape[0]))
    if self.z_scale is not None:
        z *= self.z_scale
    k = rng.integers(-5, 5, size=(N, h_dense.shape[0], 3))
    jax_seed = rng.integers(1, 100)
    return noise2samples(z, k, A_dense, h_dense, jax_seed)


def relative_error(a, b):
    return np.sqrt(((a-b)**2).sum())/(np.sqrt((a**2).sum())+np.sqrt((b**2).sum()))


@export
class Linear(objax.nn.Linear):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin, repout, extend):
        self.extend = extend
        if extend:
            self.mask, self.value = extend_dim_mask(repin)
        nin, nout = repin.size(), repout.size()
        super().__init__(nin, nout)
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w = TrainVar(orthogonal((nout, nin)))
        self.rep_W = rep_W = repout*repin.T

        rep_bias = repout
        self.Pw = rep_W.equivariant_projector()
        self.Pb = rep_bias.equivariant_projector()

        self.repin = repin
        self.repout = repout

    def __call__(self, x):  # (cin) -> (cout)
        if self.extend:
            x = x*self.mask.reshape(1, -1) + self.value.reshape(1, -1)
        # G = SE3()
        # G_sample = G.sample()

        def model(x):
            W = (self.Pw@self.w.value.reshape(-1)).reshape(*self.w.value.shape)
            b = self.Pb@self.b.value
            out = x@W.T+b
            return out

        # def rel_err(a, b):
        #     return jnp.sqrt(((a-b)**2).sum())/(jnp.sqrt((a**2).sum())+jnp.sqrt((b**2).sum()))
        # if x.shape[-1] > 12:
        #     out1 = model(self.repin.rho(G_sample)@x[0])
        #     out2 = self.repout.rho(G_sample)@model(x[0])
        #     print(rel_err(out1, out2))
        return model(x)


@export
class EMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """

    def __init__(self, rep_in, rep_out, extend):
        super().__init__()
        self.linear = Linear(rep_in, nn.gated(rep_out), extend)
        self.bilinear = nn.BiLinear(nn.gated(rep_out), nn.gated(rep_out))
        self.nonlinearity = nn.GatedNonlinearity(rep_out)

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


def uniform_rep_general(ch, *rep_types):
    """ adds all combinations of (powers of) rep_types up to
        a total size of ch channels. """
    raise NotImplementedError


@export
class EMLP(Module, metaclass=Named):
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

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3, extend=False):  # @
        super().__init__()
        logging.info("Initing EMLP (objax)")
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)

        self.G = group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            if extend:
                middle_layers = num_layers * \
                    [next(uniform_reps(ch, (group,), 2))]
            else:
                middle_layers = num_layers*[nn.uniform_rep(ch, group)]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(group)]
        else:
            if extend:
                middle_layers = [(c(group) if isinstance(
                    c, Rep) else next(uniform_reps(ch, (group,), 2))) for c in ch]
            else:
                middle_layers = [(c(group) if isinstance(c, Rep)
                                  else nn.uniform_rep(c, group)) for c in ch]
        #assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[EMLPBlock(rin, rout, extend)
              for rin, rout in zip(reps, reps[1:])],
            Linear(reps[-1], self.rep_out, extend)
        )

    def __call__(self, x, training=True):
        return self.network(x)


def swish(x):
    return jax.nn.sigmoid(x)*x


def MLPBlock(cin, cout, extend):
    # ,nn.BatchNorm0D(cout,momentum=.9),swish)#,
    return nn.Sequential(objax.nn.Linear(cin, cout), swish)


@export
class MLP(Module, metaclass=Named):
    """ Standard baseline MLP. Representations and group are used for shapes only. """

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3, extend=False):
        super().__init__()
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = nn.Sequential(
            *[MLPBlock(cin, cout, extend) for cin, cout in zip(chs, chs[1:])],
            objax.nn.Linear(chs[-1], cout)
        )

    def __call__(self, x, training=True):
        y = self.net(x)
        return y


class MixedLinear(Module):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin, repout, extend):
        self.extend = extend
        if extend:
            self.mask, self.value = extend_dim_mask(repin)
        nin, nout = repin.size(), repout.size()
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w_equiv = TrainVar(orthogonal((nout, nin)))
        self.rep_W = repout << repin

        self.w_basic = TrainVar(self.w_equiv.value*RPP_SCALE)
        self.b_basic = TrainVar(self.b.value*RPP_SCALE)
        # the bias vector has representation repout
        self.Pb = repout.equivariant_projector()
        self.Pw = self.rep_W.equivariant_projector()

    def __call__(self, x):
        if self.extend:  # e.g. slice(3, len(x), 4)
            x = x*self.mask.reshape(1, -1)+self.value.reshape(1, -1)
        W = (self.Pw@self.w_equiv.value.reshape(-1)
             ).reshape(*self.w_equiv.value.shape)
        b = self.Pb@self.b.value
        return x@(W.T + self.w_basic.value.T)+b+self.b_basic.value


def swish(x):
    return jax.nn.sigmoid(x)*x


class RPPGatedNonlinearity(Module):
    def __init__(self, rep):
        super().__init__()
        self.rep = rep
        self.w_gated = TrainVar(jnp.ones(self.rep.size())*RPP_SCALE)

    def __call__(self, values):
        gate_scalars = values[..., nn.gate_indices(self.rep)]
        gated_activations = jax.nn.sigmoid(
            gate_scalars) * values[..., :self.rep.size()]
        return gated_activations+self.w_gated.value*swish(values[..., :self.rep.size()])


class MixedEMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """

    def __init__(self, rep_in, rep_out, gnl, extend):
        super().__init__()
        self.mixedlinear = MixedLinear(rep_in, nn.gated(rep_out), extend)
        self.bilinear = nn.BiLinear(nn.gated(rep_out), nn.gated(rep_out))
        self.nonlinearity = RPPGatedNonlinearity(
            rep_out) if not gnl else nn.GatedNonlinearity(rep_out)

    def __call__(self, x):
        lin = self.mixedlinear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


@export
class MixedEMLP(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3, gnl=False, extend=False):
        super().__init__()
        logging.info("Initing EMLP")
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)

        self.G = group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            if extend:
                middle_layers = num_layers * \
                    [next(uniform_reps(ch, (group,), 2))]
            else:
                middle_layers = num_layers*[nn.uniform_rep(ch, group)]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(group)]
        else:
            if extend:
                middle_layers = [(c(group) if isinstance(
                    c, Rep) else next(uniform_reps(ch, (group,), 2))) for c in ch]
            else:
                middle_layers = [(c(group) if isinstance(c, Rep)
                                  else nn.uniform_rep(c, group)) for c in ch]

        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        # logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[MixedEMLPBlock(rin, rout, gnl, extend)
              for rin, rout in zip(reps, reps[1:])],
            MixedLinear(reps[-1], self.rep_out, extend)
        )

    def __call__(self, S, training=True):
        return self.network(S)


class MixedEMLPH(MixedEMLP):
    """ Equivariant EMLP modeling a Hamiltonian for HNN. Same args as EMLP"""
    # __doc__ += EMLP.__doc__.split('.')[1]

    def H(self, x):  # ,training=True):
        y = self.network(x)
        return y.sum()

    def __call__(self, x):
        return self.H(x)


class MixedMLPLinear(Module):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin, repout):
        nin, nout = repin.size(), repout.size()
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w_equiv = TrainVar(orthogonal((nout, nin)))
        self.rep_W = repout << repin

        self.w_basic = TrainVar(self.w_equiv.value*RPP_SCALE)
        self.b_basic = TrainVar(self.b.value*RPP_SCALE)
        # the bias vector has representation repout
        self.Pb = repout.equivariant_projector()
        self.Pw = self.rep_W.equivariant_projector()

        self.w_basic1 = TrainVar(self.w_equiv.value*RPP_SCALE)
        self.b_basic1 = TrainVar(self.b.value*RPP_SCALE)

    def __call__(self, x):
        W = (self.Pw@self.w_equiv.value.reshape(-1)
             ).reshape(*self.w_equiv.value.shape)
        b = self.Pb@self.b.value
        WoutT = W.T + self.w_basic.value.T + self.w_basic1.value.T
        bout = b + self.b_basic.value + self.b_basic1.value
        return x@WoutT+bout


class MixedGroupLinear(Module):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin, repout, repin1, repout1):
        nin, nout = repin.size(), repout.size()
        nin1, nout1 = repin1.size(), repout1.size()
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.b1 = TrainVar(objax.random.uniform((nout1,))/jnp.sqrt(nout1))
        self.w_equiv = TrainVar(orthogonal((nout, nin)))
        self.w_equiv1 = TrainVar(orthogonal((nout1, nin1)))
        self.rep_W = repout << repin
        self.rep_W1 = repout1 << repin1

        self.w_basic = TrainVar(self.w_equiv.value*RPP_SCALE)
        self.b_basic = TrainVar(self.b.value*RPP_SCALE)
        # the bias vector has representation repout
        self.Pb = repout.equivariant_projector()
        self.Pw = self.rep_W.equivariant_projector()
        self.Pb1 = repout1.equivariant_projector()
        self.Pw1 = self.rep_W1.equivariant_projector()

    def __call__(self, x):
        W = (self.Pw@self.w_equiv.value.reshape(-1)
             ).reshape(*self.w_equiv.value.shape)
        W1 = (self.Pw1@self.w_equiv1.value.reshape(-1)
              ).reshape(*self.w_equiv1.value.shape)
        b = self.Pb@self.b.value
        b1 = self.Pb1@self.b1.value
        WoutT = W.T + W1.T + self.w_basic.value.T
        bout = b + b1 + self.b_basic.value
        return x@WoutT+bout


class MixedGroup2Linear(Module):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin, repout, repin1, repout1, repin2, repout2):
        nin, nout = repin.size(), repout.size()
        nin1, nout1 = repin1.size(), repout1.size()
        nin2, nout2 = repin2.size(), repout2.size()
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.b1 = TrainVar(objax.random.uniform((nout1,))/jnp.sqrt(nout1))
        self.b2 = TrainVar(objax.random.uniform((nout2,))/jnp.sqrt(nout2))
        self.w_equiv = TrainVar(orthogonal((nout, nin)))
        self.w_equiv1 = TrainVar(orthogonal((nout1, nin1)))
        self.w_equiv2 = TrainVar(orthogonal((nout2, nin2)))
        self.rep_W = repout << repin
        self.rep_W1 = repout1 << repin1
        self.rep_W2 = repout2 << repin2

        self.w_basic = TrainVar(self.w_equiv.value*RPP_SCALE)
        self.b_basic = TrainVar(self.b.value*RPP_SCALE)
        # the bias vector has representation repout
        self.Pb = repout.equivariant_projector()
        self.Pw = self.rep_W.equivariant_projector()
        self.Pb1 = repout1.equivariant_projector()
        self.Pw1 = self.rep_W1.equivariant_projector()
        self.Pb2 = repout2.equivariant_projector()
        self.Pw2 = self.rep_W2.equivariant_projector()

    def __call__(self, x):
        W = (self.Pw@self.w_equiv.value.reshape(-1)
             ).reshape(*self.w_equiv.value.shape)
        W1 = (self.Pw1@self.w_equiv1.value.reshape(-1)
              ).reshape(*self.w_equiv1.value.shape)
        W2 = (self.Pw2@self.w_equiv2.value.reshape(-1)
              ).reshape(*self.w_equiv2.value.shape)
        b = self.Pb@self.b.value
        b1 = self.Pb1@self.b1.value
        b2 = self.Pb2@self.b2.value
        WoutT = W.T + W1.T + W2.T + self.w_basic.value.T
        bout = b + b1 + b2 + self.b_basic.value
        return x@WoutT+bout


class MixedMLPEMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """

    def __init__(self, rep_in, rep_out):
        super().__init__()
        self.mixedlinear = MixedMLPLinear(rep_in, nn.gated(rep_out))
        self.bilinear = nn.BiLinear(nn.gated(rep_out), nn.gated(rep_out))
        self.nonlinearity = RPPGatedNonlinearity(rep_out)

    def __call__(self, x):
        lin = self.mixedlinear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


class MixedGroupEMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """

    def __init__(self, rep_in, rep_out, rep_in1, rep_out1):
        super().__init__()
        self.mixedlinear = MixedGroupLinear(
            rep_in, nn.gated(rep_out), rep_in1, nn.gated(rep_out1))
        self.bilinear = nn.BiLinear(nn.gated(rep_out), nn.gated(rep_out))
        self.nonlinearity = RPPGatedNonlinearity(rep_out)

    def __call__(self, x):
        lin = self.mixedlinear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


class MixedGroup2EMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """

    def __init__(self, rep_in, rep_out, rep_in1, rep_out1, rep_in2, rep_out2):
        super().__init__()
        self.mixedlinear = MixedGroup2Linear(
            rep_in, nn.gated(rep_out), rep_in1, nn.gated(rep_out1), rep_in2, nn.gated(rep_out2))
        self.bilinear = nn.BiLinear(nn.gated(rep_out), nn.gated(rep_out))
        self.nonlinearity = RPPGatedNonlinearity(rep_out)

    def __call__(self, x):
        lin = self.mixedlinear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


@export
class MixedMLPEMLP(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):  # @
        super().__init__()
        logging.info("Initing EMLP")
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            middle_layers = num_layers*[nn.uniform_rep(ch, group)]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(group)]
        else:
            middle_layers = [(c(group) if isinstance(c, Rep)
                              else nn.uniform_rep(c, group)) for c in ch]
        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        # logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[MixedMLPEMLPBlock(rin, rout)
              for rin, rout in zip(reps, reps[1:])],
            MixedMLPLinear(reps[-1], self.rep_out)
        )

    def __call__(self, S, training=True):
        return self.network(S)


class MixedMLPEMLPH(MixedMLPEMLP):
    """ Equivariant EMLP modeling a Hamiltonian for HNN. Same args as EMLP"""
    # __doc__ += EMLP.__doc__.split('.')[1]

    def H(self, x):  # ,training=True):
        y = self.network(x)
        return y.sum()

    def __call__(self, x):
        return self.H(x)


@export
class MixedGroupEMLP(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):  # @
        super().__init__()
        logging.info("Initing EMLP")
        self.rep_in = rep_in(group[0])
        self.rep_out = rep_out(group[0])
        self.rep_in1 = rep_in(group[1])
        self.rep_out1 = rep_out(group[1])
        self.G = group[0]
        self.G1 = group[1]
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            middle_layers = num_layers*[nn.uniform_rep(ch, group[0])]
            middle_layers1 = num_layers*[nn.uniform_rep(ch, group[1])]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(group[0])]
            middle_layers1 = num_layers*[ch(group[1])]
        else:
            middle_layers = [(c(group[0]) if isinstance(c, Rep)
                              else nn.uniform_rep(c, group[0])) for c in ch]
            middle_layers1 = [(c(group[1]) if isinstance(c, Rep)
                              else nn.uniform_rep(c, group[1])) for c in ch]
        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        reps1 = [self.rep_in1]+middle_layers1
        # logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[MixedGroupEMLPBlock(rin, rout, rin1, rout1)
              for rin, rout, rin1, rout1 in zip(reps, reps[1:], reps1, reps1[1:])],
            MixedGroupLinear(reps[-1], self.rep_out, reps1[-1], self.rep_out1)
        )

    def __call__(self, S, training=True):
        return self.network(S)


@export
class MixedGroupEMLPv2(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):  # @
        super().__init__()
        logging.info("Initing EMLP")
        G, G1 = group
        self.rep_in = rep_in(G)
        self.rep_out = rep_out(G)
        self.rep_in1 = rep_in(G1)
        self.rep_out1 = rep_out(G1)

        self.G = G
        self.G1 = G1
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            sum_rep, sum_rep1 = uniform_reps(ch, (G, G1))
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            middle_layers = num_layers*[sum_rep]
            middle_layers1 = num_layers*[sum_rep1]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(G)]
            middle_layers1 = num_layers*[ch(G1)]
        else:
            middle_layers = []
            middle_layers1 = []
            for c in ch:
                if isinstance(c, Rep):
                    middle_layers.append(c(G))
                    middle_layers1.append(c(G1))
                else:
                    sum_rep, sum_rep1 = uniform_reps(c, (G, G1))
                    middle_layers.append(sum_rep)
                    middle_layers1.append(sum_rep1)
        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        reps1 = [self.rep_in1]+middle_layers1
        # logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[MixedGroupEMLPBlock(rin, rout, rin1, rout1)
              for rin, rout, rin1, rout1 in zip(reps, reps[1:], reps1, reps1[1:])],
            MixedGroupLinear(reps[-1], self.rep_out, reps1[-1], self.rep_out1)
        )

    def __call__(self, S, training=True):
        return self.network(S)


class MixedGroupEMLPH(MixedGroupEMLP):
    """ Equivariant EMLP modeling a Hamiltonian for HNN. Same args as EMLP"""
    # __doc__ += EMLP.__doc__.split('.')[1]

    def H(self, x):  # ,training=True):
        y = self.network(x)
        return y.sum()

    def __call__(self, x):
        return self.H(x)


@export
class MixedGroup2EMLP(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):  # @
        super().__init__()
        logging.info("Initing EMLP")
        self.rep_in = rep_in(group[0])
        self.rep_out = rep_out(group[0])
        self.rep_in1 = rep_in(group[1])
        self.rep_out1 = rep_out(group[1])
        self.rep_in2 = rep_in(group[2])
        self.rep_out2 = rep_out(group[2])
        self.G = group[0]
        self.G1 = group[1]
        self.G2 = group[2]
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            middle_layers = num_layers*[nn.uniform_rep(ch, group[0])]
            middle_layers1 = num_layers*[nn.uniform_rep(ch, group[1])]
            middle_layers2 = num_layers*[nn.uniform_rep(ch, group[2])]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(group[0])]
            middle_layers1 = num_layers*[ch(group[1])]
            middle_layers2 = num_layers*[ch(group[2])]
        else:
            middle_layers = [(c(group[0]) if isinstance(c, Rep)
                              else nn.uniform_rep(c, group[0])) for c in ch]
            middle_layers1 = [(c(group[1]) if isinstance(c, Rep)
                              else nn.uniform_rep(c, group[1])) for c in ch]
            middle_layers2 = [(c(group[2]) if isinstance(c, Rep)
                              else nn.uniform_rep(c, group[2])) for c in ch]
        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        reps1 = [self.rep_in1]+middle_layers1
        reps2 = [self.rep_in2]+middle_layers2
        # logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[MixedGroup2EMLPBlock(rin, rout, rin1, rout1, rin2, rout2)
              for rin, rout, rin1, rout1, rin2, rout2 in zip(reps, reps[1:], reps1, reps1[1:], reps2, reps2[1:])],
            MixedGroup2Linear(reps[-1], self.rep_out, reps1[-1],
                              self.rep_out1, reps2[-1], self.rep_out2)
        )

    def __call__(self, S, training=True):
        return self.network(S)


@export
class MixedGroup2EMLPv2(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):  # @
        super().__init__()
        logging.info("Initing EMLP")
        self.rep_in = rep_in(group[0])
        self.rep_out = rep_out(group[0])
        self.rep_in1 = rep_in(group[1])
        self.rep_out1 = rep_out(group[1])
        self.rep_in2 = rep_in(group[2])
        self.rep_out2 = rep_out(group[2])
        self.G = group[0]
        self.G1 = group[1]
        self.G2 = group[2]
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            sum_rep, sum_rep1, sum_rep2 = uniform_reps(ch, group)
            middle_layers = num_layers*[sum_rep]
            middle_layers1 = num_layers*[sum_rep1]
            middle_layers2 = num_layers*[sum_rep2]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(group[0])]
            middle_layers1 = num_layers*[ch(group[1])]
            middle_layers2 = num_layers*[ch(group[2])]
        else:
            middle_layers = []
            middle_layers1 = []
            middle_layers2 = []
            for c in ch:
                if isinstance(c, Rep):
                    middle_layers.append(c(group[0]))
                    middle_layers1.append(c(group[1]))
                    middle_layers2.append(c(group[2]))
                else:
                    sum_rep, sum_rep1, sum_rep2 = uniform_reps(c, group)
                    middle_layers.append(sum_rep)
                    middle_layers1.append(sum_rep1)
                    middle_layers2.append(sum_rep2)
        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        reps1 = [self.rep_in1]+middle_layers1
        reps2 = [self.rep_in2]+middle_layers2
        # logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[MixedGroup2EMLPBlock(rin, rout, rin1, rout1, rin2, rout2)
              for rin, rout, rin1, rout1, rin2, rout2 in zip(reps, reps[1:], reps1, reps1[1:], reps2, reps2[1:])],
            MixedGroup2Linear(reps[-1], self.rep_out, reps1[-1],
                              self.rep_out1, reps2[-1], self.rep_out2)
        )

    def __call__(self, S, training=True):
        return self.network(S)


class WeightedLinear(Module):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin, repout, weighted):
        nin, nout = repin.size(), repout.size()
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w_equiv = TrainVar(orthogonal((nout, nin)))
        self.rep_W = repout << repin

        self.w_basic = TrainVar(self.w_equiv.value*RPP_SCALE)
        self.b_basic = TrainVar(self.b.value*RPP_SCALE)
        # the bias vector has representation repout
        self.Pb = repout.equivariant_projector()
        self.Pw = self.rep_W.equivariant_projector()
        self.weight_gate = nn.Sequential(
            objax.nn.Linear(nin, nout),
            jax.nn.sigmoid
        )
        self.weighted = weighted

    def __call__(self, x):
        alpha = self.weight_gate(x)
        W = (self.Pw@self.w_equiv.value.reshape(-1)
             ).reshape(*self.w_equiv.value.shape)
        b = self.Pb@self.b.value
        if self.weighted:
            return (1-alpha)*(x@W.T+b) + alpha*(x@self.w_basic.value.T+self.b_basic.value)
        return x@W.T+b + alpha*(x@self.w_basic.value.T+self.b_basic.value)


class WeightedEMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """

    def __init__(self, rep_in, rep_out, weighted):
        super().__init__()
        self.mixedlinear = WeightedLinear(rep_in, nn.gated(rep_out), weighted)
        self.bilinear = nn.BiLinear(nn.gated(rep_out), nn.gated(rep_out))
        self.nonlinearity = RPPGatedNonlinearity(rep_out)

    def __call__(self, x):
        lin = self.mixedlinear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


@ export
class WeightedEMLP(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3, weighted=False):  # @
        super().__init__()
        logging.info("Initing EMLP")
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)

        self.G = group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            middle_layers = num_layers*[nn.uniform_rep(ch, group)]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(group)]
        else:
            middle_layers = [(c(group) if isinstance(c, Rep)
                              else nn.uniform_rep(c, group)) for c in ch]
        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        # logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[WeightedEMLPBlock(rin, rout, weighted=weighted)
              for rin, rout in zip(reps, reps[1:])],
            WeightedLinear(reps[-1], self.rep_out, weighted=weighted)
        )

    def __call__(self, S, training=True):
        return self.network(S)


class MultiGroupLinear(objax.nn.Linear):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin, repout, repin1, repout1):
        nin, nout = repin.size(), repout.size()
        nin1, nout1 = repin1.size(), repout1.size()
        super().__init__(nin, nout)
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w = TrainVar(orthogonal((nout, nin)))
        self.rep_W = rep_W = repout*repin.T
        self.b1 = TrainVar(objax.random.uniform((nout1,))/jnp.sqrt(nout1))
        self.w1 = TrainVar(orthogonal((nout1, nin1)))
        self.rep_W1 = rep_W1 = repout1*repin1.T

        rep_bias = repout
        rep_bias1 = repout1
        self.Pw = rep_W.equivariant_projector()
        self.Pb = rep_bias.equivariant_projector()
        logging.info(f"Linear W components:{rep_W.size()} rep:{rep_W}")
        self.Pw1 = rep_W1.equivariant_projector()
        self.Pb1 = rep_bias1.equivariant_projector()

    def __call__(self, x):  # (cin) -> (cout)
        logging.debug(f"linear in shape: {x.shape}")
        W = (self.Pw@self.w.value.reshape(-1)).reshape(*self.w.value.shape)
        b = self.Pb@self.b.value
        out = x@W.T+b
        W1 = (self.Pw1@self.w1.value.reshape(-1)).reshape(*self.w1.value.shape)
        b1 = self.Pb1@self.b1.value
        out1 = x@W1.T+b1
        logging.debug(f"linear out shape:{out.shape}")
        return 0.5*out+0.5*out1


class MultiEMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """

    def __init__(self, rep_in, rep_out, rep_in1, rep_out1):
        super().__init__()
        self.linear = MultiGroupLinear(rep_in, nn.gated(
            rep_out), rep_in1, nn.gated(rep_out1))
        self.bilinear = nn.BiLinear(nn.gated(rep_out), nn.gated(rep_out))
        self.nonlinearity = nn.GatedNonlinearity(rep_out)

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


@export
class MultiEMLP(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):  # @
        super().__init__()
        logging.info("Initing MultiEMLP (objax)")
        G, G1 = group
        self.rep_in = rep_in(G)
        self.rep_out = rep_out(G)
        self.rep_in1 = rep_in(G1)
        self.rep_out1 = rep_out(G1)

        self.G = G
        self.G1 = G1
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            middle_layers = num_layers*[nn.uniform_rep(ch, G)]
            middle_layers1 = num_layers*[nn.uniform_rep(ch, G1)]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(G)]
            middle_layers1 = num_layers*[ch(G1)]
        else:
            middle_layers = [(c(G) if isinstance(c, Rep)
                              else nn.uniform_rep(c, G)) for c in ch]
            middle_layers1 = [(c(G1) if isinstance(c, Rep)
                              else nn.uniform_rep(c, G1)) for c in ch]
        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        reps1 = [self.rep_in1]+middle_layers1
        logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[MultiEMLPBlock(rin, rout, rin1, rout1) for rin, rout,
              rin1, rout1 in zip(reps, reps[1:], reps1, reps1[1:])],
            MultiGroupLinear(reps[-1], self.rep_out, reps1[-1], self.rep_out1)
        )

    def __call__(self, x, training=True):
        return self.network(x)


@export
class MultiEMLPv2(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):  # @
        super().__init__()
        logging.info("Initing MultiEMLP (objax)")
        G, G1 = group
        self.rep_in = rep_in(G)
        self.rep_out = rep_out(G)
        self.rep_in1 = rep_in(G1)
        self.rep_out1 = rep_out(G1)

        self.G = G
        self.G1 = G1
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            sum_rep, sum_rep1 = uniform_reps(ch, (G, G1))
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            middle_layers = num_layers*[sum_rep]
            middle_layers1 = num_layers*[sum_rep1]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(G)]
            middle_layers1 = num_layers*[ch(G1)]
        else:
            middle_layers = []
            middle_layers1 = []
            for c in ch:
                if isinstance(c, Rep):
                    middle_layers.append(c(G))
                    middle_layers1.append(c(G1))
                else:
                    sum_rep, sum_rep1 = uniform_reps(c, (G, G1))
                    middle_layers.append(sum_rep)
                    middle_layers1.append(sum_rep1)
        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        reps1 = [self.rep_in1]+middle_layers1
        logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[MultiEMLPBlock(rin, rout, rin1, rout1) for rin, rout,
              rin1, rout1 in zip(reps, reps[1:], reps1, reps1[1:])],
            MultiGroupLinear(reps[-1], self.rep_out, reps1[-1], self.rep_out1)
        )

    def __call__(self, x, training=True):
        return self.network(x)


@export
class ExtraGroupLinear(objax.nn.Linear):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin, repout, repin1, repout1):
        nin, nout = repin.size(), repout.size()
        super().__init__(nin, nout)
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w = TrainVar(orthogonal((nout, nin)))

        self.rep_W = rep_W = repout*repin.T
        rep_bias = repout
        for rep in rep_W.reps:
            rep.solcache = {}
        for rep in rep_bias.reps:
            rep.solcache = {}
        self.Qw = rep_W.equivariant_basis()
        self.Qb = rep_bias.equivariant_basis()
        self.Pw = rep_W.equivariant_projector()
        self.Pb = rep_bias.equivariant_projector()

        self.rep_W1 = rep_W1 = repout1*repin1.T
        rep_bias1 = repout1
        for rep in rep_W1.reps:
            rep.solcache = {}
        for rep in rep_bias1.reps:
            rep.solcache = {}
        self.Qw1 = rep_W1.equivariant_basis()
        self.Qb1 = rep_bias1.equivariant_basis()
        self.Pw1 = rep_W1.equivariant_projector()
        self.Pb1 = rep_bias1.equivariant_projector()

    def __call__(self, x):  # (cin) -> (cout)
        W = (self.Pw@self.w.value.reshape(-1)).reshape(*self.w.value.shape)
        b = self.Pb@self.b.value
        out = x@W.T+b
        return out


class ExtraGroupEMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """

    def __init__(self, rep_in, rep_out, rep_in1, rep_out1):
        super().__init__()
        self.linear = ExtraGroupLinear(
            rep_in, nn.gated(rep_out), rep_in1, nn.gated(rep_out1))
        self.bilinear = nn.BiLinear(nn.gated(rep_out), nn.gated(rep_out))
        self.nonlinearity = nn.GatedNonlinearity(rep_out)

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


@export
class ExtraGroupEMLP(Module, metaclass=Named):
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

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):  # @
        super().__init__()
        logging.info("Initing EMLP (objax)")
        group, extra_group = group
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)
        self.rep_in1 = rep_in(extra_group)
        self.rep_out1 = rep_out(extra_group)

        self.G = group
        self.G1 = extra_group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            sum_rep, sum_rep1 = uniform_reps(ch, (group, extra_group))
            middle_layers = num_layers*[sum_rep]
            middle_layers1 = num_layers*[sum_rep1]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(group)]
            middle_layers1 = num_layers*[ch(extra_group)]
        else:
            middle_layers = []
            middle_layers1 = []
            for c in ch:
                if isinstance(c, Rep):
                    middle_layers.append(c(group))
                    middle_layers1.append(c(extra_group))
                else:
                    sum_rep, sum_rep1 = uniform_reps(c, (group, extra_group))
                    middle_layers.append(sum_rep)
                    middle_layers1.append(sum_rep1)
        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        reps1 = [self.rep_in1]+middle_layers1
        logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[ExtraGroupEMLPBlock(rin, rout, rin1, rout1)
              for rin, rout, rin1, rout1 in zip(reps, reps[1:], reps1, reps1[1:])],
            ExtraGroupLinear(reps[-1], self.rep_out, reps1[-1], self.rep_out1)
        )

    def __call__(self, x, training=True):
        return self.network(x)


def uniform_reps(ch, groups, maxlim_rank=None):
    """ A heuristic method for allocating a given number of channels (ch)
        into tensor types. Attempts to distribute the channels evenly across
        the different tensor types. Useful for hands off layer construction.

        Args:
            ch (int): total number of channels
            group (Group): symmetry group
        Returns:
            SumRep: The direct sum representation with dim(V)=ch
        """
    for g in groups:
        assert groups[0].d == g.d
    d = groups[0].d
    # number of tensors of each rank
    if maxlim_rank is None:
        maxlim_rank = nn.lambertW(ch, d)
    Ns = np.zeros((maxlim_rank+1,), int)
    while ch > 0:
        # compute the max rank tensor that can fit up to
        max_rank = maxlim_rank
        Ns[:max_rank+1] += np.array([d**(max_rank-r)
                                    for r in range(max_rank+1)], dtype=int)
        ch -= (max_rank+1)*d**max_rank  # compute leftover channels
    sum_rep_list = [[] for _ in range(len(groups))]
    for r, nr in enumerate(Ns):
        allocations = binomial_allocations(nr, r, groups)
        for i, alloc in enumerate(allocations):
            sum_rep_list[i].append(alloc)
    sum_reps = [sum(sum_rep) for sum_rep in sum_rep_list]
    return (sum_rep.canonicalize()[0] for sum_rep in sum_reps)


def binomial_allocations(N, rank, groups):
    """ Allocates N of tensors of total rank r=(p+q) into
        T(k,r-k) for k=0,1,...,r to match the binomial distribution.
        For orthogonal representations there is no
        distinction between p and q, so this op is equivalent to N*T(rank)."""
    if N == 0:
        return 0
    n_binoms = N//(2**rank)
    n_leftover = N % (2**rank)
    even_split_list = [[] for _ in range(len(groups))]
    for k in range(rank+1):
        nums = n_binoms*int(binom(rank, k))
        for i, g in enumerate(groups):
            even_split_list[i].append(nums*T(k, rank-k, g))
    even_splits = [sum(even_split) for even_split in even_split_list]
    rng = np.random.default_rng(N)
    ps = rng.binomial(rank, .5, n_leftover)
    # ps = np.random.binomial(rank, .5, n_leftover)
    raggeds = [sum([T(int(p), rank-int(p), g) for p in ps]) for g in groups]
    return (even_split+ragged for even_split, ragged in zip(even_splits, raggeds))


@ export
class SoftEMLP(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, group, ch=384, num_layers=3):  # @
        super().__init__()
        logging.info("Initing EMLP")
        self.rep_in = rep_in(group)
        self.rep_out = rep_out(group)

        self.G = group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            middle_layers = num_layers*[nn.uniform_rep(ch, group)]
        elif isinstance(ch, Rep):
            middle_layers = num_layers*[ch(group)]
        else:
            middle_layers = [(c(group) if isinstance(c, Rep)
                              else nn.uniform_rep(c, group)) for c in ch]
        # assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        # logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[SoftEMLPBlock(rin, rout)
              for rin, rout in zip(reps, reps[1:])],
            SoftLinear(reps[-1], self.rep_out)
        )

    def __call__(self, S, training=True):
        return self.network(S)


class SoftLinear(Module):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin, repout):
        nin, nout = repin.size(), repout.size()
        init_b = objax.random.uniform((nout,))/jnp.sqrt(nout)
        init_w = orthogonal((nout, nin))
        self.b = TrainVar(init_b)
        self.w = TrainVar(init_w)
        self.rep_W = repout << repin

        self.Pb = repout.equivariant_projector()
        self.Pw = self.rep_W.equivariant_projector()

        self.b_std = jnp.std(init_b)
        self.w_std = jnp.std(init_w)

    def __call__(self, x):
        return x@self.w.value.T+self.b.value


class SoftEMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """

    def __init__(self, rep_in, rep_out):
        super().__init__()
        self.linear = SoftLinear(rep_in, nn.gated(rep_out))
        self.bilinear = nn.BiLinear(nn.gated(rep_out), nn.gated(rep_out))
        self.nonlinearity = RPPGatedNonlinearity(rep_out)

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


@export
class SoftMultiEMLP(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, groups,
                 ch=384, num_layers=3, gnl=False, rpp_init=False, extend=False):
        super().__init__()
        logging.info("Initing SoftMultiEMLP")
        self.rep_in_list = [rep_in(g) for g in groups]
        self.rep_out_list = [rep_out(g) for g in groups]
        self.groups = groups
        if isinstance(ch, int):
            if extend:
                middle_layers_list = [num_layers*[sum_rep]
                                      for sum_rep in uniform_reps(ch, groups, 2)]
            else:
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
                    if extend:
                        for i, sum_rep in enumerate(uniform_reps(c, groups, 2)):
                            middle_layers_list[i].append(num_layers*[sum_rep])
                    else:
                        for i, sum_rep in enumerate(uniform_reps(c, groups)):
                            middle_layers_list[i].append(num_layers*[sum_rep])

        reps_list = [[rep_in]+middle_layers for rep_in,
                     middle_layers in zip(self.rep_in_list, middle_layers_list)]
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
        self.network = nn.Sequential(
            *[SoftMultiEMLPBlock(rins, routs, gnl, rpp_init, extend)
              for rins, routs in zip(rin_list, rout_list)],
            SoftMultiGroupLinear(
                rout_list[-1], self.rep_out_list, rpp_init, extend)
        )

    def __call__(self, S, training=True):
        return self.network(S)

    def equiv_error(self, idx, input, n_transforms, forward=None):
        if forward is None:
            forward = self.network
        g = self.groups[idx]
        g.rng_samples = types.MethodType(rng_samples, g)
        rep_in = self.rep_in_list[idx]
        rep_out = self.rep_out_list[idx]
        input_transforms = (rep_in.rho(s)
                            for s in g.rng_samples(n_transforms, seed=0))
        output_transforms = (rep_out.rho(s)
                             for s in g.rng_samples(n_transforms, seed=0))
        trans_input_list = [(T@input.transpose()).transpose()
                            for T in input_transforms]
        out1 = [forward(trans_input) for trans_input in trans_input_list]
        output = forward(input)
        out2 = [(T@output.transpose()).transpose() for T in output_transforms]
        errors = [relative_error(o1, o2) for o1, o2 in zip(out1, out2)]
        return sum(errors)/len(errors)


class SoftMultiEMLPBlock(Module):
    def __init__(self, rep_in_list, rep_out_list, gnl, rpp_init, extend):
        super().__init__()
        self.linear = SoftMultiGroupLinear(
            rep_in_list, [nn.gated(rep_out) for rep_out in rep_out_list], rpp_init, extend)
        self.bilinear = nn.BiLinear(
            nn.gated(rep_out_list[0]), nn.gated(rep_out_list[0]))
        self.nonlinearity = RPPGatedNonlinearity(
            rep_out_list[0]) if not gnl else nn.GatedNonlinearity(rep_out_list[0])

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin) + lin
        return self.nonlinearity(preact)


class SoftMultiGroupLinear(Module):
    def __init__(self, repin_list, repout_list, rpp_init, extend):
        self.extend = extend
        if extend:
            self.mask, self.value = extend_dim_mask(repin_list[0])
        nin, nout = repin_list[0].size(), repout_list[0].size()
        init_b = objax.random.uniform((nout,))/jnp.sqrt(nout)
        init_w = orthogonal((nout, nin))

        rep_W_list = [repout << repin for repout,
                      repin in zip(repout_list, repin_list)]
        rep_bias_list = repout_list
        self.Pw_list = []
        self.Pb_list = []
        self.Qw_list = []
        self.Qb_list = []
        for rep_W, rep_bias in zip(rep_W_list, rep_bias_list):
            rep_W = reset_solcache(rep_W)
            rep_bias = reset_solcache(rep_bias)
            self.Qw_list.append(rep_W.equivariant_basis())
            self.Pw_list.append(rep_W.equivariant_projector())
            self.Qb_list.append(rep_bias.equivariant_basis())
            self.Pb_list.append(rep_bias.equivariant_projector())

        if rpp_init:
            init_w = (self.Pw_list[0]@init_w.reshape(-1)
                      ).reshape(*init_w.shape) + RPP_SCALE*init_w
            init_b = self.Pb_list[0]@init_b + RPP_SCALE*init_b

        self.b = TrainVar(init_b)
        self.w = TrainVar(init_w)

    def __call__(self, x):
        if self.extend:
            x = x*self.mask.reshape(1, -1) + self.value.reshape(1, -1)
        return x@self.w.value.T + self.b.value


@export
class SoftMixedEMLP(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, groups,
                 ch=384, num_layers=3, gnl=False, rpp_init=False, extend=False):
        super().__init__()
        logging.info("Initing SoftMultiEMLP")
        self.rep_in_list = [rep_in(g) for g in groups]
        self.rep_out_list = [rep_out(g) for g in groups]
        self.groups = groups
        if isinstance(ch, int):
            if extend:
                middle_layers_list = [num_layers*[sum_rep]
                                      for sum_rep in uniform_reps(ch, groups, 2)]
            else:
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
                    if extend:
                        for i, sum_rep in enumerate(uniform_reps(c, groups, 2)):
                            middle_layers_list[i].append(num_layers*[sum_rep])
                    else:
                        for i, sum_rep in enumerate(uniform_reps(c, groups)):
                            middle_layers_list[i].append(num_layers*[sum_rep])

        reps_list = [[rep_in]+middle_layers for rep_in,
                     middle_layers in zip(self.rep_in_list, middle_layers_list)]
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
        self.network = nn.Sequential(
            *[SoftMixedEMLPBlock(rins, routs, gnl, rpp_init, extend)
              for rins, routs in zip(rin_list, rout_list)],
            SoftMixedGroupLinear(
                rout_list[-1], self.rep_out_list, rpp_init, extend)
        )
        self.original_projections = None

    def __call__(self, S, training=True):
        return self.network(S)

    def get_proj(self, idx):
        projections = []
        for lyr in self.network:
            if isinstance(lyr, SoftMixedEMLPBlock):
                lyr = lyr.linear
            if idx == 0:
                Pw = lyr.Pw
                Pb = lyr.Pb
            else:
                Pw = lyr.Pw_list[idx-1]
                Pb = lyr.Pb_list[idx-1]
            projections.append((Pw, Pb))
        return projections

    def set_proj(self, projections):
        if self.original_projections is None:
            self.original_projections = self.get_proj(0)
        for lyr, (Pw, Pb) in zip(self.network, projections):
            if isinstance(lyr, SoftMixedEMLPBlock):
                lyr = lyr.linear
            lyr.Pw = Pw
            lyr.Pb = Pb

    def reset_proj(self):
        self.set_proj(self.original_projections)
        self.original_projections = None


class SoftMixedEMLPBlock(Module):
    def __init__(self, rep_in_list, rep_out_list, gnl, rpp_init, extend):
        super().__init__()
        self.linear = SoftMixedGroupLinear(
            rep_in_list, [nn.gated(rep_out) for rep_out in rep_out_list], rpp_init, extend)
        self.bilinear = nn.BiLinear(
            nn.gated(rep_out_list[0]), nn.gated(rep_out_list[0]))
        self.nonlinearity = RPPGatedNonlinearity(
            rep_out_list[0]) if not gnl else nn.GatedNonlinearity(rep_out_list[0])

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin) + lin
        return self.nonlinearity(preact)


def reset_solcache(reps):
    if isinstance(reps, ScalarRep) or isinstance(reps, Base):
        reps.solcache = {}
    else:
        for rep in reps.reps:
            rep.solcache = {}
    return reps


class SoftMixedGroupLinear(Module):
    def __init__(self, repin_list, repout_list, rpp_init, extend):
        self.extend = extend
        if extend:
            self.mask, self.value = extend_dim_mask(repin_list[0])
        nin, nout = repin_list[0].size(), repout_list[0].size()
        init_b = objax.random.uniform((nout,))/jnp.sqrt(nout)
        init_w = orthogonal((nout, nin))

        print(repin_list[0])
        rep_W_list = [repout << repin for repout,
                      repin in zip(repout_list, repin_list)]
        rep_bias_list = repout_list
        self.Pw_list = []
        self.Pb_list = []
        for i, (rep_W, rep_bias) in enumerate(zip(rep_W_list, rep_bias_list)):
            rep_W = reset_solcache(rep_W)
            rep_bias = reset_solcache(rep_bias)
            if i == 0:
                self.Pw = rep_W.equivariant_projector()
                self.Pb = rep_bias.equivariant_projector()
            else:
                self.Pw_list.append(rep_W.equivariant_projector())
                self.Pb_list.append(rep_bias.equivariant_projector())

        if rpp_init:
            init_w = (self.Pw_list[-1]@init_w.reshape(-1)
                      ).reshape(*init_w.shape) + RPP_SCALE*init_w
            init_b = (self.Pb_list[-1]@init_b.reshape(-1)
                      ).reshape(*init_b.shape) + RPP_SCALE*init_b

        self.b = TrainVar(init_b)
        self.w = TrainVar(init_w)

    def __call__(self, x):
        if self.extend:
            x = x*self.mask.reshape(1, -1) + self.value.reshape(1, -1)
        W = (self.Pw@self.w.value.reshape(-1)).reshape(*self.w.value.shape)
        b = self.Pb@self.b.value
        out = x@W.T + b
        return out


class MixedLinearV2(Module):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin_list, repout_list, extend):
        self.extend = extend
        if extend:
            self.mask, self.value = extend_dim_mask(repin_list[0])
        nin, nout = repin_list[0].size(), repout_list[0].size()
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w_equiv = TrainVar(orthogonal((nout, nin)))
        rep_W_list = [repout << repin for repout,
                      repin in zip(repout_list, repin_list)]
        rep_bias_list = repout_list
        self.Pw_list = []
        self.Pb_list = []
        for i, (rep_W, rep_bias) in enumerate(zip(rep_W_list, rep_bias_list)):
            rep_W = reset_solcache(rep_W)
            rep_bias = reset_solcache(rep_bias)
            if i == 0:
                self.Pw = rep_W.equivariant_projector()
                self.Pb = rep_bias.equivariant_projector()
            else:
                self.Pw_list.append(rep_W.equivariant_projector())
                self.Pb_list.append(rep_bias.equivariant_projector())

        self.w_basic = TrainVar(self.w_equiv.value*RPP_SCALE)
        self.b_basic = TrainVar(self.b.value*RPP_SCALE)

    def __call__(self, x):
        if self.extend:  # e.g. slice(3, len(x), 4)
            x = x*self.mask.reshape(1, -1)+self.value.reshape(1, -1)
        W = (self.Pw@self.w_equiv.value.reshape(-1)
             ).reshape(*self.w_equiv.value.shape)
        b = self.Pb@self.b.value
        return x@(W.T + self.w_basic.value.T)+b+self.b_basic.value


class MixedEMLPBlockV2(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """

    def __init__(self, rep_in_list, rep_out_list, gnl, extend):
        super().__init__()
        self.linear = MixedLinearV2(
            rep_in_list, [nn.gated(rep_out) for rep_out in rep_out_list], extend)
        self.bilinear = nn.BiLinear(
            nn.gated(rep_out_list[0]), nn.gated(rep_out_list[0]))
        self.nonlinearity = RPPGatedNonlinearity(
            rep_out_list[0]) if not gnl else nn.GatedNonlinearity(rep_out_list[0])

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)


@export
class MixedEMLPV2(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, groups, ch=384, num_layers=3, gnl=False, extend=False):
        super().__init__()
        logging.info("Initing EMLP")
        self.rep_in_list = [rep_in(g) for g in groups]
        self.rep_out_list = [rep_out(g) for g in groups]

        self.groups = groups
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            if extend:
                middle_layers_list = [num_layers*[sum_rep]
                                      for sum_rep in uniform_reps(ch, groups, 2)]
            else:
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
                    if extend:
                        for i, sum_rep in enumerate(uniform_reps(c, groups, 2)):
                            middle_layers_list[i].append(num_layers*[sum_rep])
                    else:
                        for i, sum_rep in enumerate(uniform_reps(c, groups)):
                            middle_layers_list[i].append(num_layers*[sum_rep])

        reps_list = [[rep_in]+middle_layers for rep_in,
                     middle_layers in zip(self.rep_in_list, middle_layers_list)]
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
        self.network = nn.Sequential(
            *[MixedEMLPBlockV2(rins, routs, gnl, extend)
              for rins, routs in zip(rin_list, rout_list)],
            MixedLinearV2(
                rout_list[-1], self.rep_out_list, extend)
        )

    def __call__(self, S, training=True):
        return self.network(S)

    def equiv_error(self, idx, input, n_transforms, forward=None):
        if forward is None:
            forward = self.network
        g = self.groups[idx]
        g.rng_samples = types.MethodType(rng_samples, g)
        rep_in = self.rep_in_list[idx]
        rep_out = self.rep_out_list[idx]
        input_transforms = (rep_in.rho(s)
                            for s in g.rng_samples(n_transforms, seed=0))
        output_transforms = (rep_out.rho(s)
                             for s in g.rng_samples(n_transforms, seed=0))
        trans_input_list = [(T@input.transpose()).transpose()
                            for T in input_transforms]
        out1 = [forward(trans_input) for trans_input in trans_input_list]
        output = forward(input)
        out2 = [(T@output.transpose()).transpose() for T in output_transforms]
        errors = [relative_error(o1, o2) for o1, o2 in zip(out1, out2)]
        return sum(errors)/len(errors)


@export
class HybridSoftEMLP(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, groups,
                 ch=384, num_layers=3, gnl=False, rpp_init=False, extend=False):
        super().__init__()
        logging.info("Initing SoftMultiEMLP")
        self.rep_in_list = [rep_in(g) for g in groups]
        self.rep_out_list = [rep_out(g) for g in groups]
        self.groups = groups
        if isinstance(ch, int):
            if extend:
                middle_layers_list = [num_layers*[sum_rep]
                                      for sum_rep in uniform_reps(ch, groups, 2)]
            else:
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
                    if extend:
                        for i, sum_rep in enumerate(uniform_reps(c, groups, 2)):
                            middle_layers_list[i].append(num_layers*[sum_rep])
                    else:
                        for i, sum_rep in enumerate(uniform_reps(c, groups)):
                            middle_layers_list[i].append(num_layers*[sum_rep])

        reps_list = [[rep_in]+middle_layers for rep_in,
                     middle_layers in zip(self.rep_in_list, middle_layers_list)]
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
        self.network = nn.Sequential(
            *[HybridSoftEMLPBlock(rins, routs, gnl, rpp_init, extend)
              for rins, routs in zip(rin_list, rout_list)],
            HybridSoftLinear(
                rout_list[-1], self.rep_out_list, rpp_init, extend)
        )

    def __call__(self, S, training=True):
        return self.network(S)

    def set_state(self, state):
        for lyr in self.network:
            if isinstance(lyr, HybridSoftEMLPBlock):
                lyr = lyr.linear
            lyr.state = state

    def get_current_state(self):
        lyr = self.network[0].linear
        return lyr.state


class HybridSoftEMLPBlock(Module):
    def __init__(self, rep_in_list, rep_out_list, gnl, rpp_init, extend):
        super().__init__()
        self.linear = HybridSoftLinear(
            rep_in_list, [nn.gated(rep_out) for rep_out in rep_out_list], rpp_init, extend)
        self.bilinear = nn.BiLinear(
            nn.gated(rep_out_list[0]), nn.gated(rep_out_list[0]))
        self.nonlinearity = RPPGatedNonlinearity(
            rep_out_list[0]) if not gnl else nn.GatedNonlinearity(rep_out_list[0])

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin) + lin
        return self.nonlinearity(preact)


class HybridSoftLinear(Module):
    def __init__(self, repin_list, repout_list, rpp_init, extend):
        self.extend = extend
        if extend:
            self.mask, self.value = extend_dim_mask(repin_list[0])
        nin, nout = repin_list[0].size(), repout_list[0].size()
        init_b = objax.random.uniform((nout,))/jnp.sqrt(nout)
        init_w = orthogonal((nout, nin))

        rep_W_list = [repout << repin for repout,
                      repin in zip(repout_list, repin_list)]
        rep_bias_list = repout_list
        self.Pw_list = []
        self.Pb_list = []
        for rep_W, rep_bias in zip(rep_W_list, rep_bias_list):
            rep_W = reset_solcache(rep_W)
            rep_bias = reset_solcache(rep_bias)
            self.Pw_list.append(rep_W.equivariant_projector())
            self.Pb_list.append(rep_bias.equivariant_projector())

        if rpp_init:
            init_w = (self.Pw_list[0]@init_w.reshape(-1)
                      ).reshape(*init_w.shape) + RPP_SCALE*init_w
            init_b = (self.Pb_list[0]@init_b.reshape(-1)
                      ).reshape(*init_b.shape) + RPP_SCALE*init_b

        self.b = TrainVar(init_b)
        self.w = TrainVar(init_w)

        self.state = -1

    def __call__(self, x):
        if self.extend:
            x = x*self.mask.reshape(1, -1) + self.value.reshape(1, -1)
        if self.state == -1:  # only regularization
            return x@self.w.value.T + self.b.value
        else:  # subgroup projection
            Pw = self.Pw_list[self.state]
            Pb = self.Pb_list[self.state]
            W = (Pw@self.w.value.reshape(-1)).reshape(*self.w.value.shape)
            b = Pb@self.b.value
            return x@W.T + b


@export
class BiEMLP(Module, metaclass=Named):

    def __init__(self, rep_in, rep_out, groups, ch=384, num_layers=3, gnl=False, extend=False):
        super().__init__()
        assert len(groups) == 2
        self.rep_in_list = [rep_in(g) for g in groups]
        self.rep_out_list = [rep_out(g) for g in groups]

        self.groups = groups
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch, int):
            # [uniform_rep(ch,group) for _ in range(num_layers)]
            if extend:
                middle_layers_list = [num_layers*[sum_rep]
                                      for sum_rep in uniform_reps(ch, groups, 2)]
            else:
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
                    if extend:
                        for i, sum_rep in enumerate(uniform_reps(c, groups, 2)):
                            middle_layers_list[i].append(num_layers*[sum_rep])
                    else:
                        for i, sum_rep in enumerate(uniform_reps(c, groups)):
                            middle_layers_list[i].append(num_layers*[sum_rep])

        reps_list = [[rep_in]+middle_layers for rep_in,
                     middle_layers in zip(self.rep_in_list, middle_layers_list)]
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
        self.network = nn.Sequential(
            *[BiEMLPBlock(rins, routs, gnl, extend)
              for rins, routs in zip(rin_list, rout_list)],
            BiEMLPLinear(
                rout_list[-1], self.rep_out_list, extend)
        )

    def __call__(self, S, training=True):
        return self.network(S)


class BiEMLPLinear(Module):
    """ Basic equivariant Linear layer from repin to repout."""

    def __init__(self, repin_list, repout_list, extend):
        self.extend = extend
        if extend:
            self.mask, self.value = extend_dim_mask(repin_list[0])
        nin, nout = repin_list[0].size(), repout_list[0].size()
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w_equiv = TrainVar(orthogonal((nout, nin)))
        rep_W_list = [repout << repin for repout,
                      repin in zip(repout_list, repin_list)]
        rep_bias_list = repout_list
        self.Pw_list = []
        self.Pb_list = []
        for i, (rep_W, rep_bias) in enumerate(zip(rep_W_list, rep_bias_list)):
            rep_W = reset_solcache(rep_W)
            rep_bias = reset_solcache(rep_bias)
            if i == 0:
                self.Pw = rep_W.equivariant_projector()
                self.Pb = rep_bias.equivariant_projector()
            else:
                self.Pw_list.append(rep_W.equivariant_projector())
                self.Pb_list.append(rep_bias.equivariant_projector())

        self.w_basic = TrainVar(self.w_equiv.value*RPP_SCALE)
        self.b_basic = TrainVar(self.b.value*RPP_SCALE)

    def __call__(self, x):
        if self.extend:  # e.g. slice(3, len(x), 4)
            x = x*self.mask.reshape(1, -1)+self.value.reshape(1, -1)
        W1 = (self.Pw@self.w_equiv.value.reshape(-1)
              ).reshape(*self.w_equiv.value.shape)
        b1 = self.Pb@self.b.value
        W2 = (self.Pw_list[0]@self.w_basic.value.reshape(-1)
              ).reshape(*self.w_basic.value.shape)
        b2 = self.Pb_list[0]@self.b_basic.value
        return x@(W1.T + W2.T)+b1+b2


class BiEMLPBlock(Module):
    def __init__(self, rep_in_list, rep_out_list, gnl, extend):
        super().__init__()
        self.linear = BiEMLPLinear(
            rep_in_list, [nn.gated(rep_out) for rep_out in rep_out_list], extend)
        self.bilinear = nn.BiLinear(
            nn.gated(rep_out_list[0]), nn.gated(rep_out_list[0]))
        self.nonlinearity = RPPGatedNonlinearity(
            rep_out_list[0]) if not gnl else nn.GatedNonlinearity(rep_out_list[0])

    def __call__(self, x):
        lin = self.linear(x)
        preact = self.bilinear(lin)+lin
        return self.nonlinearity(preact)
