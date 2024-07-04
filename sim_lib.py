from model import *
from allele import *
import numpy as np
import time
import random

def simShell(tmax: float, mdls: list[SIR], nt: float=2e5, alleles: list[allele]=[]):
    '''
    Manages the time iterations of the simulation.

    ### Parameters
    tmax: The maximum amount of time to run the simulation for.
    mdls: A tuple containing the models governing each population (initialised with parameters & populations).
    nt: The number of time steps to use, as a 'float' (e.g. 2e5 - integer in floating-point form).
    alleles: The list of all possible alleles (allele objects). Irrelevant if not using the allele model.

    ### Returns
    ts: A NumPy array of the times visited by the simulation (not equispaced nor integer).
    ps: A NumPy array that contains the populations (flattened) at each time. Rows index population, columns index time.
    times: An array containing the computation times of the various components of the method.
    pops: A list of all population objects.
    '''
    dt = tmax/(nt - 1)
    nt = int(nt)

    if len(alleles):
        loci = ''
        for a in alleles:
            if a.locus not in loci: loci += a.locus
        genotypes = ['' for i in range(3**len(loci))]
        for l in loci:
            for i in range(len(genotypes)):
                if i%3 == 0: genotypes[i] += (l.upper() + l.lower())
                if i%3 == 1: genotypes[i] += 2*l.upper()
                if i%3 == 2: genotypes[i] += 2*l.lower()
                genotypes[i] += '.'
            genotypes.sort()
        genotypes = [g[:-1] for g in genotypes]
        new_models = []
        for m in mdls:
            for g in genotypes:
                new_model = m.updateGenotype(g, alleles)
                if 'init' in new_model.pop.inf.keys():
                    new_model.pop.addPop(new_model.pop.getPop(), g)
                    del new_model.pop.inf['init']
                    del new_model.pop.rec['init']
                new_models += [new_model]
        mdls = new_models

    [m.setRs() for m in mdls]
    ts_i = np.array(range(int(nt)))
    ps = np.empty(shape=(nt, len(mdls), len(mdls[0].pop.getAllPop())))
    pops = [m.pop for m in mdls]
    pops_check = []
    for p in list(pops):
        if hex(id(p)) not in pops_check: pops_check += [hex(id(p))]
        else: pops.remove(p)
    num_pops = len(pops)
    times = [0 for i in range(16)]
    num_Rs = len(mdls[0].Rs)
    num_mdls = len(mdls)
    all_Rs = np.array([0.0 for i in range(num_mdls*num_Rs)])

    for i in ts_i:
        tm = time.time()
        for j in range(num_mdls):
            mdls[j].setRs()
            for k in range(num_Rs):
                all_Rs[j*num_Rs+k] = mdls[j].Rs[k]
        times[0] += time.time() - tm # a little expensive. scales poorly with simulation complexity
        tm = time.time()
        sum_Rs = sum(all_Rs)
        times[1] += time.time() - tm
        tm = time.time()
        if not sum_Rs: break
        times[2] += time.time() - tm
        tm = time.time()
        Xs = adaptSim(all_Rs/sum_Rs, sum_Rs, dt)
        times[3] += time.time() - tm # most expensive
        for i_m in range(num_mdls):
            for i_r in range(num_Rs):
                tm = time.time()
                rpt = Xs[i_m*num_Rs+i_r]
                if rpt: mdls[i_m].trans(i_r, rpt)
                times[4] += time.time() - tm # 2nd most expensive
        tm = time.time()
        for i_p in range(num_pops):
            ps[i][i_p] = pops[i_p].getAllPop()
        times[5] += time.time() - tm
        # All remaining measurement blocks are left over from a more complicated and inefficient time
        # Preserved in case they prove useful sometime in the future

        tm = time.time()
        times[6] += time.time() - tm
        tm = time.time()
        
        times[7] += time.time() - tm
        tm = time.time()

        times[8] += time.time() - tm
        tm = time.time()
        times[9] += time.time() - tm
        tm = time.time()
        times[10] += time.time() - tm
        tm = time.time()
        times[11] += time.time() - tm
        tm = time.time()
        times[12] += time.time() - tm
        tm = time.time()
        times[13] += time.time() - tm
        tm = time.time()
        times[14] += time.time() - tm
    return ts_i*dt, ps, times, pops

def adaptSim(ps: np.ndarray[float], sum_Rs: float, dt: float):
    '''
    Adaptively picks the best way to estimate the results of the model. Returns an array containing the number of times each event
    occurs (ordered according to the input probabilities).

    ### Parameters
    ps: The relative probabilities of each event, as a NumPy array.
    sum_Rs: The net rate of all events.
    dt: The size of the time step.
    '''
    Xs = 0*ps
    p_cond = 0
    rng = np.random.default_rng()
    N = int(sum_Rs*dt) # to save on time, the random variable this should technically be has been replaced with its avg value
    for i in range(len(ps)):
        if p_cond >= 1 or N <= 0: break
        p = ps[i]/(1 - p_cond)
        if p > 1: p = 1
        if p <= 0: continue
        if N > 200 and (p > 0.2 and p < 0.8): Xs[i] = int(N*p)              # Deterministic case
        elif N > 100:
            if N*p < 25 or N*(1-p) < 25: Xs[i] = rng.poisson(lam=N*p)       # Large-ish N, p close to 0 or 1
            else: Xs[i] = abs(int(rng.normal(loc=N*p, scale=N*p*(1-p))))    # Large-ish N, p close to neither 0 nor 1
        else: Xs[i] = rng.binomial(n=N, p=p)                                # Small N
        N -= Xs[i]
        p_cond += ps[i]
    return Xs