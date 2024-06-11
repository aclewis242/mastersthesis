from model import *
import numpy as np
import time
import random

def simShell(p: np.ndarray[list], tmax: float, mdls: tuple[SIR], nt: float=2e5):
    '''
    Manages the time iterations of the simulation.

    ### Parameters
    p: A tuple of 3-element lists containing the S, I, and R values of both populations.
    tmax: The maximum amount of time to run the simulation for.
    mdls: A tuple containing the models governing each population (initialised with parameters).

    ### Returns
    ts: A NumPy array of the times visited by the simulation (not equispaced nor integer).
    ps: A NumPy array that contains the populations (flattened) at each time. Rows index population, columns index time.
    ts_hs: A NumPy array containing the times at which host switch events occurred.
    '''
    # ps = np.array((p[i] for i in range(len(p)) if sum(p[i]))).flatten()
    # is_2p = bool(len(ps) - 1) # note: currently a stopgap way of dealing with the non-general pop situation; do not keep!
    dt = tmax/(nt - 1)
    nt = int(nt)
    ts_i = np.array(range(nt))
    ps = np.empty(shape=(nt, len(p.flatten())))
    ps[0] = np.array([p.flatten()])
    times = [0 for i in range(16)]
    len_vec = mdls[0].num_Es
    for i in ts_i:
        tm = time.time()
        all_Rs = np.array([mdls[i].setRs(p[i], p[i-1]) for i in [0,1]]).flatten() # note: figure out how to generalise!
        # print(f'p: {p}')
        times[0] += time.time() - tm
        tm = time.time()
        sum_Rs = sum(all_Rs)
        times[1] += time.time() - tm
        tm = time.time()
        if not sum_Rs: break
        times[2] += time.time() - tm
        tm = time.time()
        # print(f'all_Rs: {all_Rs}')
        Xs = adaptSim(all_Rs/sum_Rs, sum_Rs, dt)        
        times[3] += time.time() - tm
        is_hs = False
        # print(f'Xs: {Xs}')
        for i_x in range(2*len_vec):
            tm = time.time()
            mi = int(i_x/len_vec)
            ei = i_x%len_vec
            p_new, is_hs_new = mdls[mi].trans((p[mi], p[mi-1]), ei, Xs[i_x])
            times[4] += time.time() - tm
            tm = time.time()
            is_hs = is_hs or is_hs_new
            p = np.array([p_new[mi], p_new[mi-1]])
            times[5] += time.time() - tm
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
        # t += dt
        times[12] += time.time() - tm
        tm = time.time()
        # ps = np.vstack((ps, p.flatten()))
        # ps = np.append(ps, np.array([p.flatten()]), axis=0)
        ps[i] = p.flatten()
        times[13] += time.time() - tm
        tm = time.time()
        # ts = np.append(ts, t)
        times[14] += time.time() - tm
    tm = time.time()
    print(f'Filling out array time: {time.time()-tm}')
    return ts_i*dt, ps, times

def adaptSim(ps, sum_Rs, dt: float):
    Xs = 0*ps
    p_cond = 0
    rng = np.random.default_rng()
    # N = int(sum_Rs*dt)
    for i in range(len(ps)):
        if p_cond >= 1 or N <= 0: break
        p = ps[i]/(1 - p_cond)
        if p > 1: p = int(p)# ; print(p)
        if p < 0: p = 0
        if N > 1000 and (p > 0.2 and p < 0.8): Xs[i] = int(N*p)
        elif N > 100:
            if N*p < 25: Xs[i] = rng.poisson(lam=N*p) # print(f'N: {N}; p: {p}'); 
            elif N*(1-p) < 25: Xs[i] = N - rng.poisson(lam=N*p)# ; print(f'neg check: {Xs[i]}')
            else: Xs[i] = int(rng.normal(loc=N*p, scale=N*p*(1-p)))# ; print(f'Xs[i]: {Xs[i]}')
        else: Xs[i] = rng.binomial(n=N, p=p)
        N -= Xs[i-1]
        p_cond += ps[i]
    return Xs