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
    is_hyb = False
    Rs = []
    Es = []
    pop = [-1, -1, -1]
    num_Es = -1.0
    def __init__(self, p0, **kwargs):
        '''
        Initialises the model with the given parameters.

        ### Parameters
        bd: Birth/death rate
        ir: Infection rate
        rr: Recovery rate
        wi: Waning immunity rate
        itr: Interspecific transmission rate
        is_hyb: Whether or not the model is hybrid stochastic/deterministic
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
        E8 = [0, 0, 0]  # Interspecific transmission
        # Es_raw = [E1, E2, E3, E4, E5, E6, E7, E8]
        self.Es = [E1, E2, E3, E4, E5, E6, E7]
        # self.Es = [[E, [0, 0, 0]] for E in Es_raw]
        # m = [[0, 0, 0], [-1, 1, 0]]
        # self.Es[-1] = [list(map(add, self.Es[-1][i], m[i])) for i in (0,1)]
        # self.num_Es = len(self.Es)
        # self.Rs = [0 for i in Es_raw]
        self.setRs()

    def setRs(self):
        '''
        Generates the different transition rates based on the model parameters and the given populations.

        ### Parameters
        p1: A 3-element list containing the susceptible, infected, and recovered populations. This is for the 'main' population, i.e.
        the one governed by this model's parameter set.
        p2: As above, but for the other population.
        '''
        [S, I, R] = self.pop
        S = float(int(S))
        I = float(int(I))
        R = float(int(R))
        N = S + I + R
        if not N: return [0 for r in self.Rs]
        self.Rs = [N*self.bd,
                    self.ir*S*I/N,
                    self.rr*I,
                    self.bd*S,
                    self.bd*I,
                    self.bd*R,
                    self.wi*R] + [self.itr[p2]*I*p2.pop[0]/(N+sum(p2.pop)) for p2 in self.itr]
        # print([self.itr[p2]*I*p2.S for p2 in self.itr])
        return self.Rs

    def trans(self, idx: int, rpt: int=1):
        '''
        Effects the changes in the population dictated by the simulation.

        ### Parameters
        p: A tuple of 3-element lists containing the S, I, and R values of both populations.
        idx: The index of the desired event.
        rpt: The number of times to repeat said event.

        ### Returns
        p_new: The new population, in the same format as the input.
        is_hs: A bool for whether or not this was a host switch event.
        '''
        mdl = self
        if idx >= 7:
            mdl = list(self.itr.keys())[idx-7]
            idx = 1
        mdl.pop = list(map(add, mdl.pop, list(map(lambda x: float(rpt)*x, self.Es[idx]))))
        # is_hs = bool(self.Es[idx][1][1])
        # return rv, is_hs
        return self.pop