from selectors import EpollSelector
import torch
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp
from emlp.reps import Scalar, Vector, T
from torch.utils.data import Dataset
from oil.utils.utils import export, Named, Expression, FixedNumpySeed
from emlp.groups import SO, O, Trivial, Lorentz, RubiksCube, Cube, SL
from functools import partial
import itertools
from jax import vmap, jit
from objax import Module


@export
class RandomlyModifiedInertia(Dataset, metaclass=Named):
    def __init__(self, N=1024, k=5, noise_std=0.1, MOG=False):
        super().__init__()
        self.dim = (1+3)*k
        self.X = torch.randn(N, self.dim)
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
class ModifiedInertia(Dataset, metaclass=Named):
    def __init__(self, N=1024, k=5):
        super().__init__()
        self.dim = (1+3)*k
        self.X = torch.randn(N, self.dim)
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
        target = inertia + 3e-1*vgT
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
        self.X = torch.randn(N, self.dim)
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
        #Xstd[k:] = (np.sqrt((self.X[:,k:].reshape(N,k,3)**2).mean((0,2))[:,None]) + np.zeros((k,3))).reshape(k*3)
        Xstd[k:] = (np.abs(self.X[:, k:].reshape(N, k, 3)).mean(
            (0, 2))[:, None] + np.zeros((k, 3))).reshape(k*3)
        Ymean = 0*self.Y.mean(0)
        #Ystd = np.sqrt(((self.Y-Ymean)**2).mean((0,1)))+ np.zeros_like(Ymean)
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
        self.X = torch.randn(N, self.dim)
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
        #Xstd[k:] = (np.sqrt((self.X[:,k:].reshape(N,k,3)**2).mean((0,2))[:,None]) + np.zeros((k,3))).reshape(k*3)
        Xstd[k:] = (np.abs(self.X[:, k:].reshape(N, k, 3)).mean(
            (0, 2))[:, None] + np.zeros((k, 3))).reshape(k*3)
        Ymean = 0*self.Y.mean(0)
        #Ystd = np.sqrt(((self.Y-Ymean)**2).mean((0,1)))+ np.zeros_like(Ymean)
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
    def __init__(self, N=1024, k=3):
        super().__init__()
        self.dim = 4*k
        self.X = torch.randn(N, self.dim)
        self.X[:, ::4]
