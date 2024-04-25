from model import *
import numpy as np

def simShell(p: list, tmax: float, mdl: Model):
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
    ps = np.array([p])
    ts = np.array([0])
    while t < tmax:
        p = mdl.trans(p)
        dt = -np.log(np.random.default_rng().random())/sum(mdl.Rs)
        t += dt
        ps = np.vstack((ps, p))
        ts = np.append(ts, t)
    return ts, ps