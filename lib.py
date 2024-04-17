from model import *
import numpy as np

def simShell(p, mu, beta, gamma, tmax, mth=Model):
    t = 0
    mdl = mth(mu, beta, gamma)
    ps = np.array([p])
    ts = np.array([0])
    while t < tmax:
        p = mdl.trans(p)
        dt = -np.log(np.random.default_rng())/sum(mdl.Rs)
        t += dt
        ps = np.vstack(ps, p)
        ts = np.append(ts, t)
    return ts, ps