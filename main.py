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
    'bd': 0.0385, # corresponding to a life cycle of ~2 weeks
    'ir': 0.,
    'rr': 0,
    'wi': 0.,
    'nm': 'Vector'
}
HST1 = {        # Parameters for sustained host 1 & vector behavior
    'bd': 0.,
    'ir': 0.,
    'rr': 40.,
    'wi': 20.,
    'nm': 'Host 1 (stable)'
}
HST2 = {        # Parameters for epidemic host 2 & vector behavior
    'bd': 0.,
    'ir': 0.,
    'rr': 8e1,
    'wi': 6.,
    'nm': 'Host 2 (epidemic)'
}

# for p_fac of 5e4, nt 2e4: epidemic params are 4e3, 1e3, 7e1 for ir, rr, wi respectively (stab/epi)

PARAMS_1 = HST1
PARAMS_2 = VEC
PARAMS_3 = HST2

def run(p0: np.ndarray=np.array([[2, 1, 0], [200, 0, 0], [2, 0, 0]], dtype='float64'), p_fac: float=5e4, t_max: float=1, nt: float=5e4,
        do_par_scale: bool=False):
    '''
    Run the simulation.

    ### Parameters
    p0: The initial population ratios (S, I, R) as a NumPy array of 3-element NumPy arrays.
    p_fac: The scale factor on population.
    t_max: Maximum simulation time. Scaled according to years.
    nt: The number of time steps to use. Generally speaking, this should run roughly parallel with p_fac.
    do_par_scale: Whether or not to scale population 2's parameter set to match population 1's set. This will generally produce 'cleaner'
        results, though epidemic behavior is not likely to be seen. !! LEGACY !!
    '''
    p0 *= p_fac
    if do_par_scale: # vestigial
        scale_factor = sum(PARAMS_1.values())/sum(PARAMS_2.values())
        for k in PARAMS_2.keys(): PARAMS_2[k] *= scale_factor
    m1 = SIR(p0[0], **PARAMS_1)
    m2 = SIR(p0[1], **PARAMS_2)
    m3 = SIR(p0[2], **PARAMS_3)
    m1.itr = {m2: 500., m3: 0.} # the number represents the rate at which its model infects m1
    m2.itr = {m1: 120., m3: 100.}
    m3.itr = {m1: 0., m2: 1e4}
    t0 = time.time()
    mdls = (m1, m2, m3)
    ts, ps, times = simShell(t_max, mdls, nt)
    ex_tm = time.time() - t0
    times_norm = list(100*normalise(np.array(times)))
    print(f'Execution time: {ex_tm}')
    print('Breakdown:')
    [print(f'{i}:\t{times_norm[i]}') for i in range(len(times))]
    print(f'Extra time: {ex_tm - sum(times)}')
    ns = ['S', 'I', 'R']
    for i in range(len(ps[0])):
        [plt.plot(ts, ps[:,i][:,j], label=ns[j]) for j in range(3)]
        plt.plot(ts, sum(ps[:,i].transpose()), label='N')
        plt.title(f'{mdls[i].nm} population')
        plt.legend()
        plt.xlabel('Simulation time')
        plt.ylabel('Population')
        plt.savefig(f'{fn(mdls[i].nm)}_t{nt}.png')
        plt.show()
        plt.close()

if __name__ == '__main__':
    run()