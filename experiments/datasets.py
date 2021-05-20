import torch
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp
from emlp.reps import Scalar,Vector,T
from torch.utils.data import Dataset
from oil.utils.utils import export,Named,Expression,FixedNumpySeed
from emlp.groups import SO,O,Trivial,Lorentz,RubiksCube,Cube
from functools import partial
import itertools
from jax import vmap,jit
from objax import Module

@export
class ModifiedInertia(Dataset,metaclass=Named):
    def __init__(self,N=1024,k=5):
        super().__init__()
        self.dim = (1+3)*k
        self.X = torch.randn(N,self.dim)
        self.X[:,:k] = F.softplus(self.X[:,:k]) # Masses
        mi = self.X[:,:k]
        ri = self.X[:,k:].reshape(-1,k,3)
        I = torch.eye(3)
        r2 = (ri**2).sum(-1)[...,None,None]
        inertia = (mi[:,:,None,None]*(r2*I - ri[...,None]*ri[...,None,:])).sum(1)
        g = I[2]# z axis
        v = (inertia*g).sum(-1)
        vgT = v[:,:,None]*g[None,None,:]
        target = inertia + 3e-1*vgT
        self.Y = target.reshape(-1,9)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = T(2)
        self.symmetry = O(3)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
        self.stats =0,1,0,1#Xmean,Xstd,Ymean,Ystd

    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]

@export
class Inertia(Dataset):
    def __init__(self,N=1024,k=5):
        super().__init__()
        self.dim = (1+3)*k
        self.X = torch.randn(N,self.dim)
        self.X[:,:k] = F.softplus(self.X[:,:k]) # Masses
        mi = self.X[:,:k]
        ri = self.X[:,k:].reshape(-1,k,3)
        I = torch.eye(3)
        r2 = (ri**2).sum(-1)[...,None,None]
        inertia = (mi[:,:,None,None]*(r2*I - ri[...,None]*ri[...,None,:])).sum(1)
        self.Y = inertia.reshape(-1,9)
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
        Xstd[:k] = np.abs(self.X[:,:k]).mean(0)#.std(0)
        #Xstd[k:] = (np.sqrt((self.X[:,k:].reshape(N,k,3)**2).mean((0,2))[:,None]) + np.zeros((k,3))).reshape(k*3)
        Xstd[k:] = (np.abs(self.X[:,k:].reshape(N,k,3)).mean((0,2))[:,None] + np.zeros((k,3))).reshape(k*3)
        Ymean = 0*self.Y.mean(0)
        #Ystd = np.sqrt(((self.Y-Ymean)**2).mean((0,1)))+ np.zeros_like(Ymean)
        Ystd = np.abs(self.Y-Ymean).mean((0,1)) + np.zeros_like(Ymean)
        self.stats =0,1,0,1#Xmean,Xstd,Ymean,Ystd

    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]
    def default_aug(self,model):
        return GroupAugmentation(model,self.rep_in,self.rep_out,self.symmetry)
