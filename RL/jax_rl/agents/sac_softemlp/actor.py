from typing import Tuple

import jax
import jax.numpy as jnp

from jax_rl.agents.actor_critic_temp import ActorCriticTemp
# from jax_rl.networks import _RPPNormalTanhPolicy
from jax_rl.datasets import Batch
from jax_rl.networks.common import InfoDict, Params
from rpp.flax import _Sequential, _SoftEMLPBlock, _SoftEMLPLinear
import collections


def isDict(pars):
    return isinstance(pars, collections.abc.Mapping)


def get_l2(pars):
    basic_l2 = 0.
    equiv_l2 = 0.
    for k, v in pars.items():
        if isDict(v):
            sub_basic_l2, sub_equiv_l2 = get_l2(v)
            basic_l2 += sub_basic_l2
            equiv_l2 += sub_equiv_l2
        else:
            if k.endswith("_basic"):
                basic_l2 += (v**2).sum()
            elif k.endswith("_equiv"):
                equiv_l2 += (v**2).sum()
    return basic_l2, equiv_l2


def equiv_regularizer(equivlength, params, *models):
    rglr1_list = [0. for _ in range(equivlength)]
    rglr2 = 0

    def accum_rglrs(Plist, param, rglr1_list, rglr2):
        Pw_list,Pb_list = Plist
        W,b = param
        for i, (Pw, Pb) in enumerate(zip(Pw_list, Pb_list)):
            W1 = Pw@W
            b1 = Pb@b
            rglr1_list[i] += 0.5*((W-W1)**2).sum()+0.5*((b-b1)**2).sum()
            rglr2 += 0.5*(W**2).sum()+0.5*(b**2).sum()
        return rglr1_list, rglr2

    for i, (k, v) in enumerate(params.items()):
        f = models[i]
        if f is None:
            continue
        if v.get('w') is not None:
            Pw_list = f.Pw_list
            Pb_list = f.Pb_list
            W = v['w'].reshape(-1)
            b = v['b']
            rglr1_list, rglr2 = accum_rglrs(
                (Pw_list,Pb_list), (W,b), rglr1_list, rglr2)
        else:
            for j, (k1,v1) in enumerate(v.items()):
                if v1.get('linear') is not None:
                    f = models[i].modules[j].linear
                    v2 = v1['linear']
                else:
                    f = models[i].modules[j]
                    v2 = v1
                Pw_list = f.Pw_list
                Pb_list = f.Pb_list
                W = v2['w'].reshape(-1)
                b = v2['b']
                rglr1_list, rglr2 = accum_rglrs(
                    (Pw_list,Pb_list), (W,b), rglr1_list, rglr2)
            
        

    return rglr1_list, rglr2


def update(sac: ActorCriticTemp,
           batch: Batch,
           wd: float, equiv: list) -> Tuple[ActorCriticTemp, InfoDict]:
    rng, key = jax.random.split(sac.rng)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = sac.actor.apply({'params': actor_params}, batch.observations)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        q1, q2 = sac.critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * sac.temp() - q).mean()

        body_rpp = sac.actor.apply_fn.body_rpp if hasattr(sac.actor.apply_fn, "body_rpp") else None
        mean_head = sac.actor.apply_fn.mean_head if hasattr(sac.actor.apply_fn, "mean_head") else None
        std_head = sac.actor.apply_fn.std_head if hasattr(sac.actor.apply_fn, "std_head") else None
        rglr = 0
        rglr1_list, rglr2 = equiv_regularizer(
            len(equiv), actor_params, body_rpp, mean_head, std_head)
        for eq, rglr1 in zip(equiv, rglr1_list):
            rglr += eq*rglr1
        rglr += wd*rglr2
        actor_loss = actor_loss + rglr

        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    new_actor, info = sac.actor.apply_gradient(actor_loss_fn)

    new_sac = sac.replace(actor=new_actor, rng=rng)

    return new_sac, info
