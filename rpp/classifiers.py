from objax.module import Module
from objax import nn
from objax.variable import TrainVar
import jax.numpy as jnp
import objax

from rpp.objax import SoftMultiEMLP


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x.reshape(x.shape[0], -1)

    def __repr__(self):
        return 'Flatten()'


class MLPBlock(Module):
    def __init__(self, in_size, out_size, activation):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.activation = activation

    def __call__(self, x):
        x = self.activation(self.linear(x))
        return x


class MLPClassifier(Module):
    def __init__(self, h, w, out, activation):
        super().__init__()
        l = h*w
        self.network = nn.Sequential([
            MLPBlock(l, l, activation=activation),
            MLPBlock(l, l, activation=activation),
            MLPBlock(l, l, activation=activation),
            nn.Linear(l, out)
        ])

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.network(x)
        return x


class SoftEMLPBlock(Module):
    def __init__(self, h, w, Pw, Pb, activation):
        super().__init__()
        self.b = TrainVar(objax.random.uniform((h*w,))/jnp.sqrt(h*w))
        self.w = TrainVar(nn.init.orthogonal((h*w, h*w)))
        self.Pw = Pw
        self.Pb = Pb
        self.activation = activation
        self.height = h
        self.width = w

    def __call__(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        W = (self.Pw@self.w.value.reshape(-1)).reshape(*self.w.value.shape)
        b = self.Pb@self.b.value
        out = (x@W.T+b).reshape(batch_size, self.height, self.width)
        out = self.activation(out)
        return out


class SoftLinear(Module):
    def __init__(self, repin, repout, repin1, repout1):
        nin, nout = repin.size(), repout.size()
        super().__init__(nin, nout)
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w = TrainVar(nn.init.orthogonal((nout, nin)))

        rep_w = repout << repin
        rep_b = repout
        rep_w.solcache = {}
        rep_b.solcache = {}
        self.Pw = rep_w.equivariant_projector()
        self.Pb = rep_b.equivariant_projector()

        rep_w1 = repout1 << repin1
        rep_b1 = repout1
        rep_w1.solcache = {}
        rep_b1.solcache = {}
        self.Pw1 = rep_w1.equivariant_projector()
        self.Pb1 = rep_b1.equivariant_projector()

    def __call__(self, x):
        W = (self.Pw@self.w.value.reshape(-1)).reshape(*self.w.value.shape)
        b = self.Pb@self.b.value
        out = x@W.T+b
        return out


class SoftEMLPClassifier(Module):
    def __init__(self, size, repin, repout, out, activation, groups):
        super().__init__()
        G, G1 = groups
        h, w = size
        rep_in, rep_in1 = repin(G), repin(G1)
        rep_out, rep_out1 = repout(G), repout(G1)
        last_out, last_out1 = out(G), out(G1)
        rep_w = rep_out*rep_in.T
        rep_w1 = rep_out1*rep_in1.T
        print("rep_out size", rep_out.size())
        rep_b = rep_out
        rep_b1 = rep_out1
        self.Pw = rep_w.equivariant_projector()
        self.Pb = rep_b.equivariant_projector()
        # for rep in rep_w1.reps:
        #     rep.solcache = {}
        # for rep in rep_b1.reps:
        #     rep.solcache = {}
        rep_w1.solcache = {}
        rep_b1.solcache = {}
        self.Pw1 = rep_w1.equivariant_projector()
        self.Pb1 = rep_b1.equivariant_projector()
        self.network = nn.Sequential([
            SoftEMLPBlock(h, w, self.Pw, self.Pb, activation),
            SoftEMLPBlock(h, w, self.Pw, self.Pb, activation),
            SoftEMLPBlock(h, w, self.Pw, self.Pb, activation),
            SoftEMLPBlock(rep_out, last_out, rep_out1, last_out1),
        ])

    def __call__(self, x):
        x = self.network(x)
        return x
