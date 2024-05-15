from sim_lib import *
import matplotlib.pyplot as plt

### Parameters
PARAMS_1 = {
    'bd': 1e-5,     # Birth/death rate
    'ir': 60,       # Infection rate
    'rr': 28,       # Recovery rate
    'wi': 0.00799,  # Waning immunity rate
    'hsr': 0.003,   # Host switch rate
}
PARAMS_2 = {
    'bd': 1e-5,     # Birth/death rate
    'ir': 55,       # Infection rate
    'rr': 40,       # Recovery rate
    'wi': 0.007318, # Waning immunity rate
    'hsr': 0.00,    # Host switch rate
}

def run(mdl: Model=SIR_base, p0: tuple=([100, 50, 0], [100, 0, 0])):
    '''
    Run the simulation.

    ### Parameters
    mdl: The model to use (see model.py for options).
    p0: The initial populations (S, I, R) as a tuple.
    '''
    m1 = mdl(**PARAMS_1)
    m2 = mdl(**PARAMS_2)
    ts, ps = simShell(p0, 2000, (m1, m2))
    ns = ['S', 'I', 'R']
    for i_end in [3, 6]:
        if not sum(p0[int(i_end/3)-1]): break
        [plt.plot(ts, ps[:,i], label=ns[i%3]) for i in range(i_end-3,i_end)]
        plt.plot(ts, sum([ps[:,i] for i in range(i_end-3,i_end)]), label='N')
        plt.title(f'{mdl.nm} model: population {int(i_end/3)}')
        plt.legend()
        plt.savefig(f'{mdl.nm.lower()}_pop_{int(i_end/3)}.png')
        plt.show()
        plt.close()

if __name__ == '__main__':
    run(mdl=SIR_base)