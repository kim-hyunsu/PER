from emlp.reps import Rep
from emlp.nn import uniform_rep
from rpp.objax import uniform_reps
from rpp.flax import MixedEMLPBlock, MixedLinear, Sequential, EMLPBlock, SoftEMLPBlock, SoftEMLPLinear
from oil.utils.utils import Named, export
import logging


def parse_rep(ch, group, num_layers):
    if isinstance(ch, int):
        middle_layers = num_layers*[uniform_rep(ch, group)]
    elif isinstance(ch, Rep):
        middle_layers = num_layers*[ch(group)]
    else:
        middle_layers = [(c(group) if isinstance(c, Rep)
                          else uniform_rep(c, group)) for c in ch]
    return middle_layers


def parse_reps(ch, groups, num_layers):
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
    return middle_layers_list


@export
def HeadlessSoftEMLP(rep_in, groups, ch=384, num_layers=3, gnl=False):
    middle_layers_list = parse_reps(ch, groups, num_layers)

    reps_list = [[rep_in]+middle_layers
                 for rep_in, middle_layers in zip(rep_in, middle_layers_list)]
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
    return Sequential(*[SoftEMLPBlock(rin, rout, gnl) for rin, rout in zip(rin_list, rout_list)])


@export
def HeadlessRPPEMLP(rep_in, group, ch=384, num_layers=3):
    logging.info("Initing RPP-EMLP (flax)")
    rep_in = rep_in(group)
    reps = [rep_in]+parse_rep(ch, group, num_layers)
    logging.info(f"Reps: {reps}")
    return Sequential(*[MixedEMLPBlock(rin, rout) for rin, rout in zip(reps, reps[1:])])


def HeadlessEMLP(rep_in, group, ch=384, num_layers=3):
    logging.info("Initing RPP-EMLP (flax)")
    rep_in = rep_in(group)
    reps = [rep_in]+parse_rep(ch, group, num_layers)
    logging.info(f"Reps: {reps}")
    return Sequential(*[EMLPBlock(rin, rout) for rin, rout in zip(reps, reps[1:])])
