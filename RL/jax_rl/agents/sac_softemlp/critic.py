from typing import Tuple

import jax
import jax.numpy as jnp
from optax import apply_if_finite

from jax_rl.agents.actor_critic_temp import ActorCriticTemp
from jax_rl.agents.sac_softemlp.actor import equiv_regularizer
from jax_rl.datasets import Batch
from jax_rl.networks.common import InfoDict, Params
from rpp.flax import _Sequential, _SoftEMLPBlock, _SoftEMLPLinear

import collections


def isDict(pars):
    return isinstance(pars, collections.abs.Mapping)


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


# def equiv_regularizer(equivlength, *models):
#     rglr1_list = [0. for _ in range(equivlength)]
#     rglr2 = 0

#     def accum_rglrs(lyr, rglr1_list, rglr2):
#         Pw_list = lyr.Pw_list
#         Pb_list = lyr.Pb_list
#         W = lyr.variables['params']['w'].reshape(-1)
#         b = lyr.variables['params']['b']
#         for i, (Pw, Pb) in enumerate(zip(Pw_list, Pb_list)):
#             W1 = Pw@W
#             b1 = Pb@b
#             rglr1_list[i] += 0.5*((W-W1)**2).sum()+0.5*((b-b1)**2).sum()
#             rglr2 += 0.5*(W**2).sum()+0.5*(b**2).sum()
#         return rglr1_list, rglr2

#     for f in models:
#         if f is None:
#             continue
#         if isinstance(f, _Sequential):
#             for lyr in f.modules:
#                 if isinstance(lyr, _SoftEMLPBlock):
#                     lyr = lyr.linear
#                 rglr1_list, rglr2 = accum_rglrs(lyr, rglr1_list, rglr2)
#         else:
#             if isinstance(f, _SoftEMLPBlock):
#                 lyr = lyr.linear
#             rglr1_list, rglr2 = accum_rglrs(lyr, rglr1_list, rglr2)

#     return rglr1_list, rglr2


def target_update(sac: ActorCriticTemp, tau: float) -> ActorCriticTemp:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), sac.critic.params,
        sac.target_critic.params)

    new_target_critic = sac.target_critic.replace(params=new_target_params)

    return sac.replace(target_critic=new_target_critic)


def update(sac: ActorCriticTemp, batch: Batch, discount: float,
           soft_critic: bool, wd: float, equiv: list) -> Tuple[ActorCriticTemp, InfoDict]:
    dist = sac.actor(batch.next_observations)
    rng, key = jax.random.split(sac.rng)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_q1, next_q2 = sac.target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    if soft_critic:
        target_q -= discount * batch.masks * sac.temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = sac.critic.apply({'params': critic_params},
                                  batch.observations, batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        critic1 = sac.critic.apply_fn.critic1 if hasattr(sac.critic.apply_fn, "critic1") else None
        critic2 = sac.critic.apply_fn.critic2 if hasattr(sac.critic.apply_fn, "critic2") else None

        rglr = 0
        rglr1_list, rglr2 = equiv_regularizer(
            len(equiv), critic_params, critic1, critic2
        )
        for eq, rglr1 in zip(equiv, rglr1_list):
            rglr += eq*rglr1
        rglr += wd*rglr2

        closs = critic_loss + rglr
        return closs, {
            'critic_loss': closs,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = sac.critic.apply_gradient(critic_loss_fn)

    new_sac = sac.replace(critic=new_critic, rng=rng)

    return new_sac, info
