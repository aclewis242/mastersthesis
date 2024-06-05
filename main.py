from sim_lib import *
import matplotlib.pyplot as plt
import time

### Parameters
STAB = {     # Parameters for stable behavior (equilibrium)
    'bd': 0,        # Birth/death rate
    'ir': 60,       # Infection rate
    'rr': 28,       # Recovery rate
    'wi': 10.5,     # Waning immunity rate
    'itr': 0.1,     # Interspecific transmission rate
}
EPI = {      # Parameters for epidemic behavior (short-lived spikes of infection)
    'bd': 0,        # Birth/death rate
    'ir': 400,      # Infection rate
    'rr': 310,      # Recovery rate
    'wi': 35,       # Waning immunity rate
    'itr': 0.00,    # Interspecific transmission rate
}

PARAMS_1 = STAB
PARAMS_2 = EPI

def run(p0: np.ndarray=np.array([[2, 1, 0], [2, 0, 0]], dtype='float64'), p_fac: float=5e3, t_max: float=1, nt: float=2e5,
        do_t_scale: bool=False):
    '''
    Run the simulation.

    ### Parameters
    p0: The initial population ratios (S, I, R) as a tuple.
    p_fac: The scale factor on population.
    t_max: Maximum simulation time.
    '''
    p0 *= p_fac
    if do_t_scale: t_max *= np.log(p_fac)**1.75 + 1
    m1 = SIR(**PARAMS_1)
    m2 = SIR(**PARAMS_2)
    t0 = time.time()
    mdls = (m1, m2)
    sum_Rs_test = sum(np.array([mdls[i].setRs(p0[i], p0[i-1]) for i in [0,1]]).flatten())
    dt = t_max/(nt - 1)
    if sum_Rs_test*dt > np.log(2):
        nt = float(int(t_max/(np.log(2)/sum_Rs_test) + 1))
        print(f'NOTE: insufficient number of time steps provided. New time step count: {int(nt)}')
    ts, ps, ts_hs, times = simShell(p0, t_max, mdls, nt)
    ex_tm = time.time() - t0
    print(f'Execution time: {ex_tm}')
    print('Breakdown:')
    [print(f'{i}:\t{times[i]}') for i in range(len(times))]
    print(f'Extra time: {ex_tm - sum(times)}')
    ns = ['S', 'I', 'R']
    for i_end in [3, 6]:
        if not sum(p0[int(i_end/3)-1]): break
        [plt.plot(ts, ps[:,i], label=ns[i%3]) for i in range(i_end-3,i_end)]
        plt.plot(ts, sum([ps[:,i] for i in range(i_end-3,i_end)]), label='N')
        plt.scatter(ts_hs, 0*ts_hs, label='Interspecific transmissions', c='k')
        plt.title(f'Population {int(i_end/3)}')
        plt.legend()
        plt.xlabel('Simulation time')
        plt.ylabel('Population')
        plt.savefig(f'pop_{int(i_end/3)}.png')
        plt.show()
        plt.close()

if __name__ == '__main__':
    run()