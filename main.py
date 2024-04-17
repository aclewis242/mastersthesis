from lib import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    p0 = [100, 50, 0]                   # Initial population of susceptible, infected, recovered
    mdl = SIR_base(1e-5, 60, 28)        # Initialise the model with given parameters
    ts, ps = simShell(p0, 1000, mdl)    # Simulate
    plt.plot(ts, ps[:,0], label='S')
    plt.plot(ts, ps[:,1], label='I')
    plt.plot(ts, ps[:,2], label='R')
    plt.legend()
    plt.show()