from model import *
import numpy as np

def simShell(p: tuple[list], tmax: float, mdls: tuple[SIR]):
    '''
    Manages the time iterations of the simulation.

    ### Parameters
    p: A tuple of 3-element lists containing the S, I, and R values of both populations.
    tmax: The maximum amount of time to run the simulation for.
    mdls: A tuple containing the models governing each population (initialised with parameters).

    ### Returns
    ts: A NumPy array of the times visited by the simulation. As it uses a continuous-time Markov chain, they are neither equispaced nor integers.
    ps: A NumPy array that contains the populations (flattened) at each time. Rows contain population, columns contain time.
    ts_hs: A NumPy array containing the times at which host switch events occurred.
    '''
    t = 0
    ps = np.array([p[0]+p[1]])
    ts = np.array([0])
    ts_hs = np.array([], dtype=int)
    while t < tmax:
        all_Rs = np.array([mdls[i].setRs(p[i], p[i-1]) for i in [0,1]]).flatten()
        sum_Rs = sum(all_Rs)
        if not sum_Rs: return ts, ps
        dt = np.log(1/np.random.rand())/sum_Rs
        cmpr = np.random.rand()*sum_Rs
        sat = 0
        j = -1
        while sat <= cmpr:
            j += 1
            sat += all_Rs[j]
        len_vec = len(mdls[0].Es)
        [mi, ei] = [int(j/len_vec), int(j%len_vec)]
        p_new, is_hs = mdls[mi].trans((p[mi], p[mi-1]), ei)
        p = (p_new[mi], p_new[mi-1])
        t += dt
        ps = np.vstack((ps, p[0]+p[1]))
        ts = np.append(ts, t)
        if is_hs: ts_hs = np.append(ts_hs, t)
    return ts, ps, ts_hs