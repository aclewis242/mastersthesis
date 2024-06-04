from model import *
import numpy as np
import time

def simShell(p: np.ndarray[list], tmax: float, mdls: tuple[SIR]):
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
    t = 0
    # ps = np.array((p[i] for i in range(len(p)) if sum(p[i]))).flatten()
    ps = np.array([p.flatten()])
    # is_2p = bool(len(ps) - 1) # note: currently a stopgap way of dealing with the non-general pop situation; do not keep!
    ts = np.array([0])
    ts_hs = np.array([], dtype=int)
    times = [0 for i in range(16)]
    while t < tmax:
        tm = time.time()
        all_Rs = np.array([mdls[i].setRs(p[i], p[i-1]) for i in [0,1]]).flatten() # note: figure out how to generalise!
        times[0] += time.time() - tm
        tm = time.time()
        sum_Rs = sum(all_Rs)
        times[1] += time.time() - tm
        tm = time.time()
        if not sum_Rs: return ts, ps
        times[2] += time.time() - tm
        tm = time.time()
        dt = np.log(1/np.random.rand())/sum_Rs
        times[3] += time.time() - tm
        tm = time.time()
        cmpr = np.random.rand()*sum_Rs
        times[4] += time.time() - tm
        tm = time.time()
        sat = 0
        times[5] += time.time() - tm
        tm = time.time()
        j = -1
        times[6] += time.time() - tm
        tm = time.time()
        while sat <= cmpr:
            j += 1
            sat += all_Rs[j]
        times[7] += time.time() - tm
        tm = time.time()
        len_vec = len(mdls[0].Es)
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
        t += dt
        times[12] += time.time() - tm
        tm = time.time()
        ps = np.vstack((ps, p.flatten()))
        # ps = np.append(ps, np.array([p.flatten()]), axis=0)
        times[13] += time.time() - tm
        tm = time.time()
        ts = np.append(ts, t)
        times[14] += time.time() - tm
        tm = time.time()
        if is_hs: ts_hs = np.append(ts_hs, t)
        times[15] += time.time() - tm
    return ts, ps, ts_hs, times