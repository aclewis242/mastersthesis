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
    ts_hs = np.array([], dtype=int)
    times = [0 for i in range(16)]
    old_sum_Rs = 0
    len_vec = mdls[0].num_Es
    for i in ts_i:
        if old_sum_Rs and np.log(1/np.random.rand()) > old_sum_Rs*dt: continue
        tm = time.time()
        all_Rs = np.array([mdls[i].setRs(p[i], p[i-1]) for i in [0,1]]).flatten() # note: figure out how to generalise!
        times[0] += time.time() - tm
        tm = time.time()
        sum_Rs = sum(all_Rs)
        times[1] += time.time() - tm
        tm = time.time()
        if not sum_Rs: break
        old_sum_Rs = sum_Rs
        times[2] += time.time() - tm
        tm = time.time()
        # j = np.random.choice(range(sum([m.num_Es for m in mdls])), p=all_Rs/sum_Rs)
        j = random.choices(range(sum([m.num_Es for m in mdls])), weights=all_Rs/sum_Rs)[0]
        times[3] += time.time() - tm
        tm = time.time()
        # all_Rs/sum_Rs
        times[4] += time.time() - tm
        tm = time.time()
        # range(sum([m.num_Es for m in mdls]))
        times[5] += time.time() - tm
        tm = time.time()
        # j = -1
        times[6] += time.time() - tm
        tm = time.time()
        # while sat <= cmpr:
        #     j += 1
        #     sat += all_Rs[j]
        times[7] += time.time() - tm
        tm = time.time()
        
        times[8] += time.time() - tm
        tm = time.time()
        [mi, ei] = [int(j/len_vec), int(j%len_vec)]
        times[9] += time.time() - tm
        tm = time.time()
        p_new, is_hs = mdls[mi].trans((p[mi], p[mi-1]), ei)
        times[10] += time.time() - tm
        tm = time.time()
        p = np.array([p_new[mi], p_new[mi-1]])
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
        if is_hs: ts_hs = np.append(ts_hs, i*dt)
        times[15] += time.time() - tm
    tm = time.time()
    num_skips = 0
    for p_i in range(nt):
        if not int(sum(ps[p_i])):
            ps[p_i] = ps[p_i-1]
            num_skips += 1
    print(f'Filling out array time: {time.time()-tm}')
    print(f'Num. skips: {num_skips}')
    return ts_i*dt, ps, ts_hs, times