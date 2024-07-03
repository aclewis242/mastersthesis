from model import *
import random

MUT_PARAMS = ['ir', 'rr', 'wi', 'itr']

def makeNewStrain(mdls: list[SIR], src_sn: str='init', new_sn: str='new', is_dyn: bool=False):
    new_mdls = []
    for m in mdls:
        if m.sn == src_sn: new_mdls += [m.newStrain(new_sn, is_dyn)]
    return mdls + new_mdls

def mutateDyn(mdls: list[SIR], fav_pop: str, unf_pop: str, sn: str='new', fac: float=0.2):
    mut_fav = None
    mut_unf = None
    for m in mdls:
        # print(f'looking at strain {m.sn}, pop {m.pop.getAllPop()}')
        if m.sn == sn:
            if m.pn == fav_pop: mut_fav = m
            if m.pn == unf_pop: mut_unf = m
    p2m = random.choice(MUT_PARAMS)
    if p2m == 'rr': fac = -fac
    # if mut_fav is None or mut_unf is None: return
    mut_fav.mutate(p2m, 1+fac)
    mut_unf.mutate(p2m, 1-fac)
    # print(f'mutated parameter {p2m} ({mut_fav.pn}: {mut_fav.__dict__[p2m]}, {mut_unf.pn}: {mut_unf.__dict__[p2m]})')