from sim_lib import *
import matplotlib.pyplot as plt
import time

### Parameters
# Note: The interspecific transmission rate is in the main method itself!
STAB = {        # Parameters for stable behavior (equilibrium)
    'bd': 0,        # Birth/death rate
    'ir': 60,       # Infection rate
    'rr': 28,       # Recovery rate
    'wi': 10.5,     # Waning immunity rate
    'nm': 'Stable'
}
EPI = {         # Parameters for epidemic behavior (short-lived spikes of infection)
    'bd': 0,        # Birth/death rate
    'ir': 4e3,      # Infection rate
    'rr': 1e3,      # Recovery rate
    'wi': 7e1,       # Waning immunity rate
    'nm': 'Epidemic'
}
VEC = {         # Parameters for transmission vector behavior (mosquito)
    'bd': 1.0,  # Defined as 1 birth/death event per month
    'ir': 0.,
    'rr': 0,
    'wi': 0.,
    'pn': 'Vector',
    'sn': 'init',
}
HST1 = {        # Parameters for sustained host 1 & vector behavior
    'bd': 0.,
    'ir': 0.,
    'rr': 40.,
    'wi': 20.,
    'pn': 'Host 1 (stable)',
    'sn': 'init',
}
HST2 = {        # Parameters for extinction host 2 & vector behavior
    'bd': 0.,
    'ir': 0.,
    'rr': 5e2,
    'wi': 3.,
    'pn': 'Host 2 (extinction)',
    'sn': 'init',
}

# for p_fac of 5e4, nt 2e4: epidemic params are 4e3, 1e3, 7e1 for ir, rr, wi respectively (stab/epi)

PARAMS_1 = HST1
PARAMS_2 = VEC
PARAMS_3 = HST2

def run(p0: np.ndarray=np.array([[2, 1, 0], [200, 0, 0], [2, 0, 0]], dtype='float64'), p_fac: float=5e4, t_max: float=1., nt: float=5e4,
        plot_res: bool=True, is_dyn: bool=True, mr: float=20., t_scale: float=3.):
    '''
    Run the simulation.

    ### Parameters
    p0: The initial population ratios (S, I, R) as a NumPy array of 3-element NumPy arrays.
    p_fac: The scale factor on population.
    t_max: Maximum simulation time.
    nt: The number of time steps to use. Generally speaking, this should run roughly parallel with p_fac & t_max.
    do_par_scale: Whether or not to scale population 2's parameter set to match population 1's set. This will generally produce 'cleaner'
        results, though epidemic behavior is not likely to be seen. !! LEGACY !!
    '''
    p0 *= p_fac
    t_max *= t_scale
    nt = float(int(nt*t_scale))
    [p0_1, p0_2, p0_3] = [population(p0[i]) for i in range(3)]
    m1 = SIR(p0_1, **PARAMS_1)
    m2 = SIR(p0_2, **PARAMS_2)
    m3 = SIR(p0_3, **PARAMS_3)
    m1.itr = {p0_2: 500., p0_3: 0.} # the number represents the rate at which its model infects m1
    m2.itr = {p0_1: 120., p0_3: 100.} # temp: actually treating it as m1's pop/strain ir for relevant pop (see how this goes)
    m3.itr = {p0_1: 0., p0_2: 1e2}
    t0 = time.time()
    mdls = [m1, m2, m3]
    ts, ps, times, pops = simShell(t_max, mdls, nt, is_dyn, HST2['pn'], HST1['pn'], mr)
    ex_tm = time.time() - t0
    times_norm = list(100*normalise(np.array(times)))
    print(f'Execution time: {ex_tm}')
    print('Breakdown:')
    [print(f'{i}:\t{times_norm[i]}') for i in range(len(times))]
    print(f'Extra time: {ex_tm - sum(times)}')
    for i in range(len(mdls)):
        ns = pops[i].getAllPopNms()
        [plt.plot(ts, ps[:,i][:,j], label=ns[j]) for j in range(len(ns))]
        plt.plot(ts, sum(ps[:,i].transpose()), label='N')
        plt.title(f'{mdls[i].pn} population')
        plt.legend()
        plt.xlabel('Simulation time')
        plt.ylabel('Population')
        plt.savefig(f'{fn(mdls[i].pn)}_t{nt}.png')
        if plot_res: plt.show()
        plt.close()
    return ps

if __name__ == '__main__':
    run()