from model import *
import numpy as np

def simShell(p: tuple[list], tmax: float, mdls: tuple[Model]):
    '''
    Manages the time iterations of the simulation.

    ### Parameters
    p: The initial population, as a 3-element list (S, I, R).
    tmax: The maximum amount of time to run the simulation for.
    mdl: The model to use.

    ### Returns
    ts: A NumPy array of the times visited by the simulation. As it uses a continuous-time Markov chain, they are neither equispaced nor integers.
    ps: A NumPy array containing all of the population vectors at the various times.
    '''
    t = 0
    ps = np.array([p[0]+p[1]])
    ts = np.array([0])
    while t < tmax:
        [mdls[i].setRs(p[i], p[i-1]) for i in [0,1]]
        dts = [[np.random.exponential(1/m.Rs[ri]) for ri in range(len(m.Rs))] for m in mdls]
        dt = np.min(dts)
        [mi, ei] = [int(np.where(dts == dt)[i][0]) for i in [0,1]]
        p_new = mdls[mi].trans((p[mi], p[mi-1]), ei)
        p = (p_new[mi], p_new[mi-1])
        t += dt
        ps = np.vstack((ps, p[0]+p[1]))
        ts = np.append(ts, t)
    return ts, ps