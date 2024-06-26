from func_lib import *
from operator import add
import numpy as np

class SIR:
    '''
    The stochastic SIR model class.
    '''
    nm = ''
    bd = -1.0
    ir = -1.0
    rr = -1.0
    wi = -1.0
    itr = {}
    dt = 0.0
    Rs = []
    Es = []
    pop = [-1, -1, -1]
    num_Es = -1.0
    def __init__(self, p0: list[int], **kwargs):
        '''
        Initialises the model with the given parameters.

        ### Parameters
        p0: Initial population, as a 3-element list (S, I, R)
        bd: Birth/death rate
        ir: Infection rate
        rr: Recovery rate
        wi: Waning immunity rate
        itr: Interspecific transmission rates from this model to others (dict)
        '''
        self.pop = p0
        self.__dict__.update(kwargs)
        E1 = [1, 0, 0]  # Birth
        E2 = [-1, 1, 0] # Infection
        E3 = [0, -1, 1] # Recovery
        E4 = [-1, 0, 0] # Death of susceptible
        E5 = [0, -1, 0] # Death of infected
        E6 = [0, 0, -1] # Death of recovered
        E7 = [1, 0, -1] # Waning immunity
        self.Es = [E1, E2, E3, E4, E5, E6, E7]
        self.setRs()

    def setRs(self):
        '''
        Generates the different transition rates based on the model's parameters and population.
        '''
        [S, I, R] = self.pop
        S = float(int(S))
        I = float(int(I))
        R = float(int(R))
        N = S + I + R
        if not N: return [0. for r in self.Rs]
        self.Rs = [N*self.bd,
                    self.ir*S*I/N,
                    self.rr*I,
                    self.bd*S,
                    self.bd*I,
                    self.bd*R,
                    self.wi*R] + [self.itr[p2]*I*p2.pop[0]/(N+sum(p2.pop)) for p2 in self.itr]
        return self.Rs

    def trans(self, idx: int, rpt: int=1):
        '''
        Effects the changes in the population dictated by the simulation.

        ### Parameters
        idx: The index of the desired event, corresponding with the order of Rs.
        rpt: The number of times to repeat said event.
        '''
        mdl = self
        if idx >= 7:
            mdl = list(self.itr.keys())[idx-7]
            idx = 1
        mdl.pop = list(map(add, mdl.pop, list(map(lambda x: float(rpt)*x, self.Es[idx]))))
        return self.pop