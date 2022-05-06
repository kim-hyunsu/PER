# %%
import numpy as np
from scipy.linalg import expm
from oil.utils.utils import Named, export
import jax
import jax.numpy as jnp
from emlp.groups import Group


@export
class SL(Group):
    """ The special linear group SL(n) in n dimensions"""

    def __init__(self, n):
        self.lie_algebra = np.zeros((n*n-1, n, n))
        k = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # handle diag elements separately
                self.lie_algebra[k, i, j] = 1
                k += 1
        for l in range(n-1):
            self.lie_algebra[k, l, l] = 1
            self.lie_algebra[k, -1, -1] = -1
            k += 1
        super().__init__(n)


@export
class Union(Group):
    def __init__(self, *groups):
        self.lie_algebra = np.concatenate([g.lie_algebra for g in groups], 0)
        self.discrete_generators = np.concatenate(
            [g.discrete_generators for g in groups], 0)
        super().__init__(*groups)


@export
class RotationGroup(Group):
    def __init__(self, n):
        d = n+1
        so3_lie_algebra = np.zeros(((n*(n-1))//2, n, n))
        k = 0
        for i in range(n):
            for j in range(i):
                so3_lie_algebra[k, i, j] = 1
                so3_lie_algebra[k, j, i] = -1
                k += 1
        self.lie_algebra = np.zeros(((n*(n-1))//2, d, d))
        idx = slice(n)
        self.lie_algebra[:, idx, idx] = so3_lie_algebra
        super().__init__(n)


@export
class TranslationGroup(Group):
    def __init__(self, n):
        d = n+1
        self.lie_algebra = np.zeros(((n*(n-1))//2, d, d))
        for i in range(n):
            self.lie_algebra[i, -1, i] = 1
        super().__init__(n)


@export
def SE3():
    return Union(RotationGroup(3), TranslationGroup(3))
