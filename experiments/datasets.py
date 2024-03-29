import torch
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp
from emlp.reps import Scalar, Vector, T
from torch.utils.data import Dataset
from oil.utils.utils import export, Named, Expression, FixedNumpySeed
from emlp.groups import SO, O, Trivial, Lorentz, RubiksCube, Cube, Scaling, S
from models.groups import SE3, Union
from functools import partial
import itertools
from jax import vmap, jit
from objax import Module
import functools
from oil.utils.utils import imap
import jax

#https://github.com/kim-hyunsu/PER/blob/463c955545cb054baa87ebf44d321c99e0d6b05b/experiments/trainer/utils.py 
def minibatch_to(mb):
    try:
        if isinstance(mb,np.ndarray):
            return jax.device_put(mb)
        return jax.device_put(mb.numpy())
    except AttributeError:
        if isinstance(mb,dict):
            return type(mb)(((k,minibatch_to(v)) for k,v in mb.items()))
        else:
            return type(mb)(minibatch_to(elem) for elem in mb)

def LoaderTo(loader):
    return imap(functools.partial(minibatch_to),loader)

@export
class RandomlyModifiedInertia(Dataset, metaclass=Named):
    def __init__(self, N=1024, k=5, noise_std=0.1, MOG=False):
        super().__init__()
        self.dim = (1+3)
        rng = torch.Generator()
        rng.manual_seed(N+k+self.dim)
        self.X = torch.randn(N, self.dim, generator=rng)
        self.X[:, :k] = F.softplus(self.X[:, :k])  # Masses
        mi = self.X[:, :k]
        ri = self.X[:, k:].reshape(-1, k, 3)
        I = torch.eye(3)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] *
                   (r2*I - ri[..., None]*ri[..., None, :])).sum(1)
        g = I[2]  # z axis
        v = (inertia*g).sum(-1)
        vgT = v[:, :, None]*g[None, None, :]
        length = inertia.size(0)
        if not MOG:
            alpha = torch.normal(torch.tensor(
                0.3), torch.tensor(noise_std), size=(length, 1, 1))
        else:
            u = torch.rand(length, 1, 1)
            alpha = torch.normal(torch.tensor(0.3), torch.tensor(
                noise_std), size=(length, 1, 1))
            beta = torch.normal(torch.tensor(0.6), torch.tensor(
                noise_std), size=(length, 1, 1))
            alpha[u < 0.4] = beta[u < 0.4]
        target = inertia + alpha*vgT
        self.Y = target.reshape(-1, 9)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = T(2)
        self.symmetry = O(3)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
        self.stats = 0, 1, 0, 1  # Xmean,Xstd,Ymean,Ystd

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]


@export
class ModifiedInertiaDeprecated(Dataset, metaclass=Named):
    def __init__(self, N=1024, k=5, noise=0.3, axis=2, bias=0):
        super().__init__()
        self.k = k
        self.noise = noise
        self.axis = axis
        self.dim = (1+3)*k
        rng = torch.Generator()
        rng.manual_seed(N+k+self.dim+int(bias))
        self.X = torch.randn(N, self.dim, generator=rng)
        self.X[:, :k] = F.softplus(self.X[:, :k])  # Masses
        self.X[:, k:] = self.X[:, k:] + bias
        mi = self.X[:, :k]
        ri = self.X[:, k:].reshape(-1, k, 3)
        I = torch.eye(3)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] *
                   (r2*I - ri[..., None]*ri[..., None, :])).sum(1)
        g = I[axis]  # z axis
        v = (inertia*g).sum(-1)
        vgT = v[:, :, None]*g[None, None, :]
        target = inertia + noise*vgT
        self.Y = target.reshape(-1, 9)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = T(2)
        self.symmetry = O(3)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
        self.stats = 0, 1, 0, 1  # Xmean,Xstd,Ymean,Ystd

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def __call__(self, x):  # jax.numpy
        # x: (batch_size, x_features)
        mi = x[:, :self.k]
        ri = x[:, self.k:].reshape(-1, self.k, 3)
        I = jnp.eye(3)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] *
                   (r2*I-ri[..., None]*ri[..., None, :])).sum(1)
        g = I[self.axis]
        v = (inertia*g).sum(-1)
        vgT = v[:, :, None]*g[None, None, :]
        target = inertia+self.noise*vgT
        return target

    def default_aug(self, model):
        return GroupAugmentation(model, self.rep_in, self.rep_out, self.symmetry)


@export
class SoftModifiedInertia(Dataset, metaclass=Named):
    def __init__(self, N=1024, k=5, noise=0.3):
        super().__init__()
        self.dim = (1+3)*k
        rng = torch.Generator()
        rng.manual_seed(N+k+self.dim)
        self.X = torch.randn(N, self.dim, generator=rng)
        self.X[:, :k] = F.softplus(self.X[:, :k])  # Masses
        mi = self.X[:, :k]
        ri = self.X[:, k:].reshape(-1, k, 3)
        I = torch.eye(3)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] *
                   (r2*I - ri[..., None]*ri[..., None, :])).sum(1)

        g1 = I[0]
        v1 = (inertia*g1).sum(-1)
        vgT1 = v1[:, :, None]*g1[None, None, :]
        g2 = I[1]
        v2 = (inertia*g2).sum(-1)
        vgT2 = v2[:, :, None]*g2[None, None, :]
        g3 = I[2]
        v3 = (inertia*g3).sum(-1)
        vgT3 = v3[:, :, None]*g3[None, None, :]

        target = inertia + 0.5*noise*vgT1 + noise*vgT2 + 1.5*noise*vgT3
        self.Y = target.reshape(-1, 9)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = T(2)
        self.symmetry = O(3)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
        self.stats = 0, 1, 0, 1  # Xmean,Xstd,Ymean,Ystd

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def default_aug(self, model):
        return GroupAugmentation(model, self.rep_in, self.rep_out, self.symmetry)


@export
class NoisyModifiedInertia(Dataset):
    def __init__(self, N=1024, k=5, noise_std=0.05):
        super().__init__()
        self.dim = (1+3)*k
        rng = torch.Generator()
        rng.manual_seed(N+k+self.dim)
        self.X = torch.randn(N, self.dim, generator=rng)
        self.X[:, :k] = F.softplus(self.X[:, :k])  # Masses
        mi = self.X[:, :k]
        ri = self.X[:, k:].reshape(-1, k, 3)
        I = torch.eye(3)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] *
                   (r2*I - ri[..., None]*ri[..., None, :])).sum(1)
        self.Y = inertia.reshape(-1, 9)
        self.X[:, :k] = self.X[:, :k]*torch.exp(noise_std*torch.randn(N, k))
        self.X[:, k:] = self.X[:, k:] + noise_std*torch.randn(N, self.dim-k)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = T(2)
        self.symmetry = O(3)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
        # One has to be careful computing offset and scale in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0)
        Xmean[k:] = 0
        Xstd = np.zeros_like(Xmean)
        Xstd[:k] = np.abs(self.X[:, :k]).mean(0)  # .std(0)
        # Xstd[k:] = (np.sqrt((self.X[:,k:].reshape(N,k,3)**2).mean((0,2))[:,None]) + np.zeros((k,3))).reshape(k*3)
        Xstd[k:] = (np.abs(self.X[:, k:].reshape(N, k, 3)).mean(
            (0, 2))[:, None] + np.zeros((k, 3))).reshape(k*3)
        Ymean = 0*self.Y.mean(0)
        # Ystd = np.sqrt(((self.Y-Ymean)**2).mean((0,1)))+ np.zeros_like(Ymean)
        Ystd = np.abs(self.Y-Ymean).mean((0, 1)) + np.zeros_like(Ymean)
        self.stats = 0, 1, 0, 1  # Xmean,Xstd,Ymean,Ystd

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]


@export
class Inertia(Dataset):
    def __init__(self, N=1024, k=5):
        super().__init__()
        self.dim = (1+3)*k
        rng = torch.Generator()
        rng.manual_seed(N+k+self.dim)
        self.X = torch.randn(N, self.dim, generator=rng)
        self.X[:, :k] = F.softplus(self.X[:, :k])  # Masses
        mi = self.X[:, :k]
        ri = self.X[:, k:].reshape(-1, k, 3)
        I = torch.eye(3)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] *
                   (r2*I - ri[..., None]*ri[..., None, :])).sum(1)
        self.Y = inertia.reshape(-1, 9)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = T(2)
        self.symmetry = O(3)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
        # One has to be careful computing offset and scale in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0)
        Xmean[k:] = 0
        Xstd = np.zeros_like(Xmean)
        Xstd[:k] = np.abs(self.X[:, :k]).mean(0)  # .std(0)
        # Xstd[k:] = (np.sqrt((self.X[:,k:].reshape(N,k,3)**2).mean((0,2))[:,None]) + np.zeros((k,3))).reshape(k*3)
        Xstd[k:] = (np.abs(self.X[:, k:].reshape(N, k, 3)).mean(
            (0, 2))[:, None] + np.zeros((k, 3))).reshape(k*3)
        Ymean = 0*self.Y.mean(0)
        # Ystd = np.sqrt(((self.Y-Ymean)**2).mean((0,1)))+ np.zeros_like(Ymean)
        Ystd = np.abs(self.Y-Ymean).mean((0, 1)) + np.zeros_like(Ymean)
        self.stats = 0, 1, 0, 1  # Xmean,Xstd,Ymean,Ystd

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def default_aug(self, model):
        return GroupAugmentation(model, self.rep_in, self.rep_out, self.symmetry)


class GroupAugmentation(Module):
    def __init__(self, network, rep_in, rep_out, group):
        super().__init__()
        self.rep_in = rep_in
        self.rep_out = rep_out
        self.G = group
        self.rho_in = jit(vmap(self.rep_in.rho))
        self.rho_out = jit(vmap(self.rep_out.rho))
        self.model = network

    def __call__(self, x, training=True):
        if training:
            gs = self.G.samples(x.shape[0])
            rhout_inv = jnp.linalg.inv(self.rho_out(gs))
            return (rhout_inv@self.model((self.rho_in(gs)@x[..., None])[..., 0], training)[..., None])[..., 0]
        else:
            return self.model(x, False)


@export
class SyntheticSE3Dataset(Dataset, metaclass=Named):
    def __init__(self, N=1024, k=3, for_mlp=False, noisy=False, noise=0, sym="", shift=0, sign=0):
        super().__init__()
        d = 3
        self.N = N
        self.d = d
        self.k = k
        self.sym = sym
        self.dim = (d+1)*k
        self.noisy = noisy
        self.noise = noise
        rng = torch.Generator()
        rng.manual_seed(N+k+self.dim+int(shift))
        self.X = torch.randn(N, self.dim, generator=rng)
        self.X = self.X + shift
        if sign != 0:
            self.X = sign*torch.abs(self.X)
        self.X[:, d::(d+1)] = 1  # every 4th element

        if noisy:
            rng = torch.Generator()
            rng.manual_seed(N+k)
            epsilon = noise*torch.rand(N, 1, generator=rng)
        else:
            epsilon = noise
        X, Y = self.get_YandErr(self.X, noise=epsilon)
        if for_mlp:
            self.X = torch.cat(X, axis=1)
        self.Y = Y
        self.rep_in = k*Vector
        self.rep_out = Scalar
        self.symmetry = SE3()
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()

    def get_YandErr(self, _X, noise=0, _jax=False):
        begin = 0
        X = []
        for i in range(self.k):
            end = begin + self.d
            X.append(_X[:, begin:end])
            begin = end+1
        Y = 0
        if not _jax:
            for i in range(self.k-1):
                Y += torch.sqrt(((X[i]-X[i+1])**2).sum(1))
            Y += torch.sqrt(((X[-1]-X[0])**2).sum(1))
            Y /= self.k
        else:
            for i in range(self.k-1):
                Y += jnp.sqrt(((X[i]-X[i+1])**2).sum(1))
            Y += jnp.sqrt(((X[-1]-X[0])**2).sum(1))
            Y /= self.k

        if not _jax:
            Y = Y.unsqueeze(-1)
        else:
            Y = jnp.expand_dims(Y, -1)

        err = 0
        if self.sym == "test":
            Y = 0
            if not _jax:
                Y += (X[0]-X[1]).abs()[:, 0]
                err = torch.zeros_like(Y)
                Y = Y.unsqueeze(-1)
            else:
                Y += jnp.abs(X[0]-X[1])[:, 0]
                err = jnp.zeros_like(Y)
                Y = jnp.expand_dims(Y, -1)

        elif self.sym == "distance":
            # Y = 0
            # if not _jax:
            #     for i in range(self.k-1):
            #         Y += (X[i]-X[i+1]).abs().sum(1)
            #     Y += 3*(X[-1]-X[0]).abs().sum(1)
            #     err = torch.zeros_like(Y)
            # else:
            #     for i in range(self.k-1):
            #         Y += jnp.abs(X[i]-X[i+1]).sum(1)
            #     Y += 3*jnp.abs(X[-1]-X[0]).sum(1)
            #     err = jnp.zeros_like(Y)
            # if not _jax:
            #     Y = Y.unsqueeze(-1)
            # else:
            #     Y = jnp.expand_dims(Y, -1)
            if not _jax:
                err += torch.zeros_like(Y).squeeze(-1)
            else:
                err += jnp.squeeze(jnp.zeros_like(Y), -1)

        elif self.sym == "ball":
            # if not _jax:
            #     for i in range(self.k):
            #         err += torch.sqrt((X[i]**2).sum(1))
            #     err /= self.k
            # else:
            #     for i in range(self.k):
            #         err += jnp.sqrt((X[i]**2).sum(1))
            #     err /= self.k
            if not _jax:
                err += torch.sqrt((X[0]**2).sum(1))
            else:
                err += jnp.sqrt((X[0]**2).sum(1))

        elif self.sym == "l1distance":
            if not _jax:
                err += (X[0]-X[1]).abs()[:, 0]
            else:
                err += jnp.abs(X[0]-X[1])[:, 0]

        elif self.sym == "r3":  # r3 perfect, t3 soft
            for i in range(self.k):
                err += (X[i]**2).sum(1)

        elif self.sym == "t3":  # t3 perfect, r3 soft
            if not _jax:
                for i in range(self.k-1):
                    err += (X[i]-X[i+1]).abs().sum(1)
                err += 3*(X[-1]-X[0]).abs().sum(1)
            else:
                for i in range(self.k-1):
                    err += jnp.abs(X[i]-X[i+1]).sum(1)
                err += 3*jnp.abs(X[-1]-X[0]).sum(1)

        elif self.sym == "r3t3soft":
            # print('r3t3soft dataset')
            for i in range(self.k):
                err += (X[i]**2).sum(1)

            if not _jax:
                for i in range(self.k-1):
                    err += (X[i]-X[i+1]).abs().sum(1)
                err += 3*(X[-1]-X[0]).abs().sum(1)
            else:
                for i in range(self.k-1):
                    err += jnp.abs(X[i]-X[i+1]).sum(1)
                err += 3*jnp.abs(X[-1]-X[0]).sum(1)

        elif self.sym == "se3":  # perfect
            if not _jax:
                err = torch.zeros_like(Y.squeeze(-1))
            else:
                err = jnp.zeros_like(jnp.squeeze(Y, -1))

        elif self.sym == "rxy2":  # rxy2 perfect, t3 soft, r3soft
            # for i in range(k):
            #     err += X[i][:,:2].pow(2).sum(1)
            for i in range(self.k-1):
                err += (X[i][:, :2]**2).sum(1)
            err += 3*(X[2][:, :2]**2).sum(1)

        # elif sym == "ryz2": #ryz2 perfect, t3 soft, r3soft
        #     for i in range(k):
        #         err += X[i][:,1:3].pow(2).sum(1)

        # elif sym == "rxz2": #rxz2 perfect, t3 soft, r3soft
        #     for i in range(k):
        #         err += X[i][:,::2].pow(2).sum(1)

        elif self.sym == "txy2":  # txy2 perfect, t3 soft, r3soft
            if not _jax:
                for i in range(self.k-1):
                    err += (X[i][:, :2]-X[i+1][:, :2]).abs().sum(1)
                err += 3*(X[-1][:, :2]-X[0][:, :2]).abs().sum(1)
            else:
                for i in range(self.k-1):
                    err += jnp.abs(X[i][:, :2]-X[i+1][:, :2]).sum(1)
                err += 3*jnp.abs(X[-1][:, :2]-X[0][:, :2]).sum(1)

        # elif sym == "tyz2": #tyz2 perfect, t3 soft, r3soft
        #     for i in range(k-1):
        #         err += (X[i][:,1:3]-X[i+1][:,1:3]).abs().sum(1)
        #     err += 3*(X[-1][:,1:3]-X[0][:,1:3]).abs().sum(1)

        # elif sym == "txz2": #txz2 perfect, t3 soft, r3soft
        #     for i in range(k-1):
        #         err += (X[i][:,::2]-X[i+1][:,::2]).abs().sum(1)
        #     err += 3*(X[-1][:,::2]-X[0][:,::2]).abs().sum(1)

        if not _jax:
            err = err.unsqueeze(-1)
        else:
            err = jnp.expand_dims(err, -1)
        # err_mean = err.mean(0).unsqueeze(0)
        # err_std = err.std(0).unsqueeze(0)
        # print(f"err mean {err_mean} std {err_std}")

        Y = Y-noise*err

        return X, Y

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def __call__(self, x):
        if self.noisy:
            rng = torch.Generator()
            rng.manual_seed(self.N+self.k)
            epsilon = self.noise*torch.rand(self.N, 1, generator=rng)
        else:
            epsilon = self.noise
        _, Y = self.get_YandErr(x, noise=epsilon, _jax=True)

        return Y


@export
class ModifiedInertia(Dataset, metaclass=Named):
    def __init__(
            self, N=1024, k=5, noise=0.3, axis=2, shift=0, sign=0, dim_swap=False):
        super().__init__()
        self.k = k
        self.noise = noise
        self.axis = axis
        self.dim_swap = dim_swap
        self.dim = (1+3)*k
        rng = torch.Generator()
        rng.manual_seed(N+k+self.dim+int(shift))
        self.X = torch.randn(N, self.dim, generator=rng)
        self.X[:, :k] = F.softplus(self.X[:, :k])  # Masses
        self.X[:, k:] = self.X[:, k:] + shift
        if sign != 0:
            self.X[:, k:] = sign*torch.abs(self.X[:, k:])

        self.Y = self.func(self.X, torch)

        if not dim_swap:
            self.rep_in = k*Scalar+k*Vector
            self.rep_out = T(2)
            self.symmetry = O(3)
        else:
            self.X[:,k:] = self.X[:,k:].view(-1,k,3).transpose(1,2).reshape(-1,3*k)
            self.rep_in = 4*Vector
            self.rep_out = 9*Scalar
            self.symmetry = S(k)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
        self.stats = 0, 1, 0, 1  # Xmean,Xstd,Ymean,Ystd

    def func(self, x, op):
        mi = x[:, :self.k]
        ri = x[:, self.k:].reshape(-1, self.k, 3)
        I = op.eye(3)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] *
                   (r2*I - ri[..., None]*ri[..., None, :])).sum(1)
        if self.axis == -1:
            sum_vgT = 0
            for i in range(3):
                g = I[i]
                v = (inertia*g).sum(-1)
                vgT = v[:, :, None]*g[None, None, :]
                if i == 0:
                    sum_vgT -= vgT
                elif i == 1:
                    sum_vgT += vgT
            vgT = sum_vgT
        else:
            g = I[self.axis]  # z axis
            v = (inertia*g).sum(-1)
            vgT = v[:, :, None]*g[None, None, :]
        target = inertia + self.noise*vgT
        return target.reshape(-1, 9)

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def __call__(self, x):  # jax.numpy
        # x: (batch_size, x_features)
        return self.func(x, jnp)

    def default_aug(self, model):
        return GroupAugmentation(model, self.rep_in, self.rep_out, self.symmetry)


@export
class SyntheticRadiusDataset(Dataset, metaclass=Named):
    def __init__(self, N=1024, k=3, noise=0, sym="", shift=0, sign=0):
        super().__init__()
        self.d = 3
        self.k = k
        self.dim = self.d*k
        self.sym = sym
        self.noise = noise
        rng = torch.Generator()
        rng.manual_seed(N+k+self.dim+int(shift))
        self.X = torch.randn(N, self.dim, generator=rng)
        self.X = self.X + shift
        if sign != 0:
            self.X = sign*torch.abs(self.X)
        x = self.X.view(-1, k, self.d)
        radius = torch.sqrt((x**2).sum(2))  # SO(3) perfect
        sum_radius = radius.sum(1)
        # SO(3) perfect Scale perfect
        norm_radius = radius/sum_radius.unsqueeze(-1)
        norm_x = x.sum(1)/x.sum(2).sum(1).unsqueeze(-1)  # Scale perfect

        if sym == "so3-scale" or sym == "scale-so3":
            self.Y = norm_radius
        elif sym == "so3":
            self.Y = norm_radius - noise*radius
        elif sym == "scale":
            self.Y = norm_radius - noise*norm_x

        self.Y = self.Y.prod(1).unsqueeze(-1)

        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
        self.rep_in = k*Vector
        self.rep_out = Scalar
        self.symmetry = Union(SO(3), Scaling(3))

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def __call__(self, x):
        x = x.reshape(-1, self.k, self.d)
        radius = jnp.sqrt((x**2).sum(2))
        sum_radius = radius.sum(1)
        norm_radius = radius/jnp.expand_dims(sum_radius, -1)
        norm_x = x.sum(1)/jnp.expand_dims(x.sum(2).sum(1), -1)

        if self.sym == "so3-scale":
            y = norm_radius
        elif self.sym == "so3":
            y = norm_radius - self.noise*radius
        elif self.sym == "scale":
            y = norm_radius - self.noise*norm_x

        y = jnp.expand_dims(jnp.product(y, 1), -1)
        return y


@export
class SyntheticCosSimDataset(Dataset, metaclass=Named):
    def __init__(
            self, N=1024, k=3, noise=0, sym="", shift=0, sign=0, dim_swap=False):

        super().__init__()
        self.d = 3
        self.k = k
        self.dim = self.d*k
        self.sym = sym
        self.noise = noise
        self.dim_swap = dim_swap
        rng = torch.Generator()
        rng.manual_seed(N+k+self.dim+int(shift))
        self.X = torch.randn(N, self.dim, generator=rng)
        self.X = self.X + shift
        if sign != 0:
            self.X = sign*torch.abs(self.X)

        self.Y = self.func(self.X, torch)

        if not dim_swap:
            self.rep_in = k*Vector
            self.rep_out = Scalar
            self.symmetry = Union(SO(3), Scaling(3))
        else:
            self.X = self.X.view(-1,k,3).transpose(1,2).reshape(-1,3*k)
            self.rep_in = 3*Vector
            self.rep_out = Scalar
            self.symmetry = S(k)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()


    def func(self, x, op):
        def cosinesimilarity(a, b):
            inner = (a*b).sum(1)
            # norm_a = op.sqrt((a**2).sum(1))
            # norm_b = op.sqrt((b**2).sum(1))
            if op is jnp:
                norm_a = op.linalg.norm(a, axis=1)
                norm_b = op.linalg.norm(b, axis=1)
            elif op is torch:
                norm_a = op.norm(a, dim=1)
                norm_b = op.norm(b, dim=1)
            return inner/(norm_a*norm_b)
        if op is jnp:
            x = x.reshape(-1, self.k, self.d)
        elif op is torch:
            x = x.view(-1, self.k, self.d)
        sum_cossim = 0
        for i in range(self.k-1):
            sum_cossim += cosinesimilarity(
                x[:, i, :], x[:, i+1, :]) / self.k
        sum_cossim += cosinesimilarity(x[:, -1, :], x[:, 0, :])/self.k
        mean_radius = op.sqrt((x**2).sum(2)).mean(1)
        half = self.k//2
        fraction = op.abs(x[:, 0:half, :]).sum(2).sum(
            1)/op.abs(x[:, half:, :]).sum(2).sum(1)

        if self.sym == "so3-scale" or self.sym == "scale-so3":
            y = sum_cossim
        elif self.sym == "so3":
            y = sum_cossim - self.noise*mean_radius
        elif self.sym == "scale":
            y = sum_cossim - self.noise*fraction
        elif self.sym == "none":
            y = sum_cossim - self.noise*mean_radius + self.noise*fraction

        if op is jnp:
            y = op.expand_dims(y, -1)
        elif op is torch:
            y = y.unsqueeze(-1)

        return y

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def __call__(self, x):
        return self.func(x, jnp)
