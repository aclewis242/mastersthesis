from lib import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    p0 = [100, 0, 0]
    mth = SIR_base
    ts, ps = simShell(p0, 1, 1, 1, 100, mth)
    plt.plot(ts, ps[:,0], label='S')
    plt.plot(ts, ps[:,1], label='I')
    plt.plot(ts, ps[:,2], label='R')
    plt.show()