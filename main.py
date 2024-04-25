from sim_lib import *
import matplotlib.pyplot as plt

### Parameters
PARAMS = {
    'bd': 1e-5,     # Birth/death rate
    'ir': 60,       # Infection rate
    'rr': 28,       # Recovery rate
    'wi': 0.007318, # Waning immunity rate
}
MODELS: list[Model] = [mt(**PARAMS) for mt in MODEL_TYPES]
[BASE, WANING] = MODELS

def run(mdl: Model=BASE):
    p0 = [100, 50, 0]                   # Initial population of susceptible, infected, recovered
    ts, ps = simShell(p0, 1000, mdl)    # Simulate
    plt.plot(ts, ps[:,0], label='S')
    plt.plot(ts, ps[:,1], label='I')
    plt.plot(ts, ps[:,2], label='R')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run(WANING)