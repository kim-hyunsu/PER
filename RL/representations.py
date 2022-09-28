from emlp.reps import V,T,Rep
from emlp.groups import Z,S,SO,Group
from scipy.spatial.transform import Rotation
import jax.numpy as jnp
import numpy as np
import jax

class PseudoScalar(Rep):
    is_regular=False
    def __init__(self,G=None):
        self.G=G
        # self.concrete = (self.G is not None)
    def __call__(self,G):
        return PseudoScalar(G)
    def size(self):
        return 1
    def __str__(self):
        return "P"
    def rho(self,M):
        sign = jnp.linalg.slogdet(M@jnp.eye(M.shape[0]))[0]
        return sign*jnp.eye(1)
    def __eq__(self,other):
        return type(self)==type(other) and self.G==other.G
    def __hash__(self):
        return hash((type(self),self.G))
    @property
    def T(self):
        return self

# M = M@jnp.eye(M.shape[0])
        # i1 = (M[:2,:2]==0).all().astype(int)
        # M2 = M[:2,:2] if i1==0 else M[2:,:2]
        # i2 = M2[0,0]
        # M1 = jnp.eye(2)[[i1,1-i1]]

class SwimmerState(Rep):
    is_regular=False
    def __init__(self):
        self.G = Z(2)*Z(2)
    def __call__(self,G):
        assert G==self.G
        return self
    def size(self):
        return 10
    def __str__(self):
        return "SwimmerX"
    def rho(self,M):
        M = M@jnp.eye(M.shape[0])
        i1 = (M[:2,:2]==0).all().astype(int)
        # M2 = M[:2,:2] if i1==0 else M[2:,:2]
        M2 = jnp.where(i1==0, M[:2,:2],M[2:,:2])
        i2= M2[0,0]
        M1 = jnp.eye(2)[[i1,1-i1],:]
        # M1 = M1@jnp.eye(M1.shape[0])
        # M2 = M2@jnp.eye(M2.shape[0])
        s1  =2*M1[0,0]-1
        s2 = 2*M2[0,0]-1
        #s,p = self.S.rho(M1),self.P.rho(M1)
        s = jnp.eye(1)
        p = s1*jnp.eye(1)
        rhos = [s,p,s1*M2,s,p,s,p,s1*M2]
        return jax.scipy.linalg.block_diag(*rhos)
    def __eq__(self,other):
        return type(self)==type(other) and self.G==other.G
    def __hash__(self):
        return hash((type(self),self.G))
    @property
    def T(self):
        return self

class SwimmerAction(SwimmerState):
    def size(self):
        return 2
    def __str__(self):
        return "SwimmerA"
    def rho(self,M):
        M1,M2 = M.Ms
        M1 = M1@jnp.eye(M1.shape[0])
        M2 = M2@jnp.eye(M2.shape[0])
        s1  =2*M1[0,0]-1
        s2 = 2*M2[0,0]-1
        return s1*M2

class SwimmerActionStd(SwimmerAction):

    def size(self):
        return 2
    def __str__(self):
        return "SwimmerAstd"
    def rho(self,M):
        M1,M2 = M.Ms
        M1 = M1@jnp.eye(M1.shape[0])
        M2 = M2@jnp.eye(M2.shape[0])
        s1  =2*M1[0,0]-1
        s2 = 2*M2[0,0]-1
        return M2

class D(Group):
    def __init__(self,k):
        translation = np.eye(4)[np.array([3,0,1,2])][None]
        reflection = np.eye(4)[np.array([1,0,3,2])][None]
        self.discrete_generators = np.concatenate((translation,reflection))
        super().__init__(k)

# can this cause a bug because of pointing to the same rep object?


import numpy as np
import jax.numpy as jnp
from jax import jit
def vector_dual(v):
    v1 = jnp.stack([0*v[...,0],-v[...,2],v[...,1]],axis=-1)
    v2 = jnp.stack([v[...,2],0*v[...,1],-v[...,0]],axis=-1)
    v3 = jnp.stack([-v[...,1],v[...,0],0*v[...,2]],axis=-1)
    return jnp.stack([v1,v2,v3],-2)

def quat2rot(q):
    q = q/jnp.sqrt((q**2).sum(-1))[...,None]
    q0,v = jnp.split(q,[1],axis=-1)
    v_cross = vector_dual(v)
    R = jnp.eye(3)-2*q0[...,None]*v_cross+2*v_cross@v_cross
    return R.T



def ant_state_transform(x):
    """ Converts the quaternion in state vector to a rotation matrix"""
    z,q,angs,vcom,w,angv,forces = jnp.split(x,[1,5,13,16,19,27],axis=-1)
    R  =quat2rot(q).reshape(*q.shape[:-1],-1)
    #R = Rotation.from_quat(q).as_matrix().reshape(*q.shape[:-1],-1)
    #Rw = vector_dual(w).reshape(*w.shape[:-1],-1)
    return jnp.concatenate([z,R,angs,vcom,w,angv],-1) #remove forces

def ant_inv_state_transform(x):
    """ converts the """
    z,R,angs,vcom,w,angv,forces = jnp.split(x,[1,5+5,13+5,16+5,19+5,27+5],axis=-1)
    R = R.reshape(*R.shape[:-1],3,3)
    q = np.roll(Rotation.from_matrix(R).as_quat(),1,axis=-1)
    return jnp.concatenate([z,q,angs,vcom,w,angv,forces],-1)


ant_state_perm = np.array([0,1,2,3,4,5,7,9,11,6,8,10,12,13,14,15,16,17,18,
                            19,21,23,25,20,22,24,26])
inv_ant_state_perm = np.argsort(ant_state_perm)
ant_action_perm = np.array([0,2,4,6,1,3,5,7])
inv_ant_action_perm = np.argsort(ant_action_perm)

def ant_state_transform2(x):
    """ groups legs into 4"""
    flips = np.ones(27)
    # flips[8]*=-1
    # flips[10]*=-1
    # flips[21]*=-1
    # flips[23]*=-1
    return x[...,ant_state_perm]*flips
def ant_inv_state_transform2(x):
    y = x[...,inv_ant_state_perm]
    flips = np.ones(27)
    # flips[8]*=-1
    # flips[10]*=-1
    # flips[21]*=-1
    # flips[23]*=-1
    return y*flips

def ant_action_transform2(a):
    """ groups legs into 4 """
    #TODO: check if actions +- are also weirdly flipped
    return a[...,ant_action_perm]
def ant_inv_action_transform2(a):
    return a[...,inv_ant_action_perm]

def _walker_state_transform(x):
    """ groups left right joint correspondences together """
    (y,orient,rh,rk,ra,lh,lk,la,vcomx,\
        vcomy,angvel,vrh,vrk,vra,vlh,vlk,vla) = jnp.split(x,np.arange(1,x.shape[-1]),axis=-1)
    reordered_tuple = (y,orient,rh,lh,rk,lk,ra,la,vcomx,\
        vcomy,angvel,vrh,vlh,vrk,vlk,vra,vla)
    return jnp.concatenate(reordered_tuple,-1)

walker_perm = _walker_state_transform(np.arange(17))
inv_walker_perm = np.argsort(walker_perm)

def walker_state_transform(x):
    return x[...,walker_perm]

def inv_walker_state_transform(x):
    return x[...,inv_walker_perm]

def _inv_walker_action_transform(a):
    rh,lh,rk,lk,ra,la= jnp.split(a,np.arange(1,a.shape[-1]),axis=-1)
    return jnp.concatenate([rh,rk,ra,lh,lk,la],-1)

inv_walker_action_perm = _inv_walker_action_transform(np.arange(6))
walker_action_perm = np.argsort(inv_walker_action_perm)

def inv_walker_action_transform(a):
    return a[...,inv_walker_action_perm]

def walker_action_transform(a):
    return a[...,walker_action_perm]

def humanoid_state_transform(x):
    z,q,x_rest,extra_info = jnp.split(x,[1,5,45],axis=-1)
    R  =quat2rot(q).reshape(*q.shape[:-1],-1)
    out = jnp.concatenate([z,R,x_rest],-1)
    return out

def inv_humanoid_state_transform(x):
    z,R,x_rest,extra_info = jnp.split(x,[1,5+5,45+5],axis=-1)
    R = R.reshape(*R.shape[:-1],3,3)
    q = np.roll(Rotation.from_matrix(R).as_quat(),1,axis=-1)
    return jnp.concatenate([z,q,x_rest],-1)

leg_arm_perm = np.array([0,4,1,5,2,6,3,7,8,11,9,12,10,13])
def _humanoid_state_perm2(x):
    stuff,legsarms,vs,legsvarmsv = jnp.split(x,[13,13+14,27+6+3])
    return jnp.concatenate([stuff,legsarms[leg_arm_perm],vs,legsvarmsv[leg_arm_perm]])

humanoid_state_perm = _humanoid_state_perm2(np.arange(50))
inv_humanoid_state_perm = np.argsort(humanoid_state_perm)
humanoid_action_perm = np.concatenate([np.arange(3),3+leg_arm_perm])
inv_humanoid_action_perm = np.argsort(humanoid_action_perm)
#print(humanoid_state_perm.shape,humanoid_action_perm.shape)


def swimmer_state_transform2(x):
    """ Expands the orientations into vectors"""
    orient_vec = jnp.stack([jnp.cos(x[...,0]),jnp.sin(x[...,0])],-1)
    orient_vel = jnp.stack([-jnp.sin(x[...,0])*x[...,5],jnp.cos(x[...,0])*x[...,5]],-1)
    return jnp.concatenate([orient_vec,x[...,1:5],orient_vel,x[...,6:]],-1)

def swimmer_inv_state_transform2(x):
    """ Expands the orientations into vectors"""
    orient_angle = jnp.arctan2(x[...,1],x[...,0])[...,None]
    orient_angvel = (x[...,6]/(-x[...,1]))[...,None]#(x[...,6]**2+x[...,7]**2)
    return jnp.concatenate([orient_angle,x[...,2:6],orient_angvel,x[...,8:]],-1)

P = PseudoScalar()
vector3 = T(1)+T(0)
matrix3 = vector3**2
pseodovector3 = P*vector3
s = T(0)
# # L/R symmetries only
# environment_symmetries['Humanoid-v2'] = {
#     'state_rep':T(0)+matrix3+17*T(0)+vector3+pseodovector3+\
#             17*T(0),
#     'state_transform':humanoid_state_transform,
#     'action_rep':17*T(0),
#     'inv_action_transform':Id,
#     'symmetry_group':SO(2),
#     'action_space':"continuous",
# }
Id = lambda x:x 

#import collections
# environment_symmetries = collections.defaultdict(lambda: {
#     'state_transform':Id,
#     'inv_state_transform':Id,
#     'action_transform':Id,
#     'inv_action_transform':Id})
legarmrep = P*T(1)+3*T(1)+P*T(1)+2*T(1)
environment_symmetries={
    # 'Humanoid-v2': {
    #     'state_rep':T(0)+matrix3+17*T(0)+vector3+pseodovector3+\
    #         17*T(0),
    #     'state_transform':humanoid_state_transform,
    #     'inv_state_transform':inv_humanoid_state_transform,
    #     'action_rep':17*T(0),
    #     'action_transform':Id,
    #     'inv_action_transform':Id,
    #     'symmetry_group':SO(2),
    #     'action_space':"continuous",
    # },
    'Humanoid-v2': {
        'state_rep':s+matrix3+s+s+P+legarmrep+\
            vector3+pseodovector3+s+s+P+legarmrep,
        'state_transform':lambda x: humanoid_state_transform(x)[...,humanoid_state_perm],
        'inv_state_transform':lambda x: inv_humanoid_state_transform(x[...,inv_humanoid_state_perm]),
        'action_rep':2*s+P+legarmrep,
        'action_transform':lambda a: a[...,humanoid_action_perm],
        'inv_action_transform':lambda a: a[...,inv_humanoid_action_perm],
        'symmetry_group':Z(2),
        'action_space':"continuous",
        'middle_rep':136*T(0)+60*T(1),
    },
    # 'Ant-v2': {
    #     'state_rep':T(0)+matrix3+8*T(0)+vector3+pseodovector3+\
    #         8*T(0),#+14*vector3+14*pseodovector3,
    #     'state_transform':ant_state_transform,
    #     'inv_state_transform':ant_inv_state_transform, #TODO: write inv
    #     'action_rep':8*T(0),
    #     'action_transform':Id,
    #     'inv_action_transform':Id,
    #     'symmetry_group':SO(2),
    #     'action_space':"continuous",
    # },
    'Ant-v2': {
        'state_rep':T(0)+4*T(0)+2*T(1)+6*T(0)+\
            2*T(1),#+14*vector3+14*pseodovector3,
        'state_transform':ant_state_transform2,
        'inv_state_transform':ant_inv_state_transform2,
        'action_rep':2*T(1),
        'action_transform':ant_action_transform2,
        'inv_action_transform':ant_inv_action_transform2,
        'symmetry_group':Z(4),
        'action_space':"continuous",
        'middle_rep':136*T(0)+30*T(1),
    },
    # 'Swimmer-v2':{ # Focus just on LR symmetry now, to add front back later
    #     'state_rep':P+P+P+2*T(0)+P+P+P, # shoud vcom swap?
    #     'state_transform':Id,
    #     'inv_state_transform':Id,
    #     'action_rep':2*P,
    #     'action_std_rep':2*T(0),
    #     'action_transform':Id,
    #     'inv_action_transform':Id,
    #     'symmetry_group':Z(2),
    #     'action_space':"continuous",
    #     'middle_rep':126*T(0)+55*T(1)+5*T(2),
    # },
    'Swimmer-v2':{ # Focus just on LR symmetry now, to add front back later
        'state_rep':SwimmerState(), # shoud vcom swap?
        'state_transform':swimmer_state_transform2,
        'inv_state_transform':swimmer_inv_state_transform2,
        'action_rep':SwimmerAction(),
        'action_std_rep':SwimmerActionStd(),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2)*Z(2),
        'action_space':"continuous",
        #'middle_rep':108*T(0)+25*T(1)+3*T(2),
    },
    'Walker2d-v2':{
        'state_rep':2*T(0)+3*T(1)+3*T(0)+3*T(1),
        'state_transform':walker_state_transform,
        'inv_state_transform':inv_walker_state_transform,
        'action_rep':3*T(1),
        'action_transform':walker_action_transform,
        'inv_action_transform':inv_walker_action_transform,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        'middle_rep':136*T(0)+60*T(1),
    },
    'Hopper-v2':{ #
        'state_rep':T(0)+4*P+(P+T(0)) + 4*P, # shoud vcom swap?
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':3*P,
        'action_std_rep':3*T(0),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        'middle_rep':126*T(0)+55*T(1)+5*T(2),
    },
    'HalfCheetah-v2':{
        'state_rep':T(0) + 8*P + T(0) + 7*P,
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':6*P,
        'action_std_rep':6*T(0),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        'middle_rep':126*T(0)+55*T(1)+5*T(2),
    },
    'InclinedCartpole-v0':{
        'state_rep':P + P + P + P,
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':T(1),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"discrete",
    },
    'CartPole-v0':{
        'state_rep':P + P + P + P,
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':T(1),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"discrete",
    }
}

environment_symmetries.update({
    'HalfCheetahFull-v0':{
        'state_rep':P+T(0) + 7*P +P+ T(0) + 7*P,
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':6*P,
        'action_std_rep':6*T(0),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        #'angular_dims':[2,3,4,5,6,7,8],
    },
    'HopperFull-v0':{ #
        'state_rep':(P+T(0))+4*P+(P+T(0)) + 4*P, # shoud vcom swap?
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':3*P,
        'action_std_rep':3*T(0),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        #'angular_dims':[2,3,4,5],
    },
    'Walker2dFull-v0':{
        'state_rep':3*T(0)+3*T(1)+3*T(0)+3*T(1),
        'state_transform':walker_state_transform,
        'inv_state_transform':inv_walker_state_transform,
        'action_rep':3*T(1),
        'action_transform':walker_action_transform,
        'inv_action_transform':inv_walker_action_transform,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        #'angular_dims':[2,3,4,5,6,7,8],
    },
    'SwimmerFull-v0':{ # Focus just on LR symmetry now, to add front back later
        'state_rep':SwimmerState(), # shoud vcom swap?
        'state_transform':swimmer_state_transform2,
        'inv_state_transform':swimmer_inv_state_transform2,
        'action_rep':SwimmerAction(),
        'action_std_rep':SwimmerActionStd(),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2)*Z(2),
        'action_space':"continuous",
        #'angular_dims':[2,3,4],
    },

})
# halfcheetah_rep2 = {
#     'state_rep':2*T(0)+3*T(1)+3*T(0)+3*T(1),
#     'state_transform':walker_state_transform,
#     'action_rep':3*T(1),
#     'inv_action_transform':inv_walker_action_transform,
#     'symmetry_group':Z(2),
# }
# comment back in to use the 2nd rep version for half cheetah
# environment_symmetries['HalfCheetah-v2']=halfcheetah_rep2
