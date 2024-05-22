from sim_lib import *
import matplotlib.pyplot as plt
import time

### Parameters
STAB = {     # Parameters for stable behavior (equilibrium)
    'bd': 0,        # Birth/death rate
    'ir': 60,       # Infection rate
    'rr': 28,       # Recovery rate
    'wi': 10.5,     # Waning immunity rate
    'hsr': 0.1,     # Host switch rate
}
EPI = {      # Parameters for epidemic behavior (short-lived spikes of infection)
    'bd': 0,        # Birth/death rate
    'ir': 400,      # Infection rate
    'rr': 310,      # Recovery rate
    'wi': 35,       # Waning immunity rate
    'hsr': 0.00,    # Host switch rate
}

PARAMS_1 = STAB
PARAMS_2 = EPI

def run(mdl: Model=SIR_base, p0: tuple=([2, 1, 0], [2, 0, 0]), p_fac: float=500, t_max: float=2.4):
    '''
    Run the simulation.

    ### Parameters
    mdl: The model to use (see model.py for options).
    p0: The initial population ratios (S, I, R) as a tuple.
    p_fac: The scale factor on population.
    t_max: Maximum simulation time.
    '''
    p0 = tuple([[p0[pi][ii]*p_fac for ii in range(len(p0[pi]))] for pi in range(len(p0))])
    m1 = mdl(**PARAMS_1)
    m2 = mdl(**PARAMS_2)
    t0 = time.time()
    ts, ps, ts_hs = simShell(p0, t_max, (m1, m2))
    print(f'Execution time: {time.time()-t0}')
    ns = ['S', 'I', 'R']
    for i_end in [3, 6]:
        if not sum(p0[int(i_end/3)-1]): break
        [plt.plot(ts, ps[:,i], label=ns[i%3]) for i in range(i_end-3,i_end)]
        plt.plot(ts, sum([ps[:,i] for i in range(i_end-3,i_end)]), label='N')
        plt.scatter(ts_hs, 0*ts_hs, label='Host switches', c='k')
        plt.title(f'{mdl.nm} model: population {int(i_end/3)}')
        plt.legend()
        plt.savefig(f'{mdl.nm.lower()}_pop_{int(i_end/3)}.png')
        plt.show()
        plt.close()

if __name__ == '__main__':
    run(mdl=SIR_waning)