from func_lib import *
import numpy as np

class Model:
    '''
    The "base" form of the model. Should not really be used directly.
    '''
    bd = -1.0
    ir = -1.0
    rr = -1.0
    hsr = -1.0
    Rs = []
    Es = []
    nm = 'Model'

    def __init__(self, **kwargs):
        '''
        Initialises the model with the given parameters.

        ### Parameters
        bd: Birth/death rate
        ir: Infection rate
        rr: Recovery rate
        hsr: Host switch rate
        '''
        self.__dict__.update(kwargs)
        E1 = [1, 0, 0]
        E2 = [-1, 1, 0]
        E3 = [0, -1, 1]
        E4 = [-1, 0, 0]
        E5 = [0, -1, 0]
        E6 = [0, 0, -1]
        E7 = [0, 0, 0]
        Es_raw = [E1, E2, E3, E4, E5, E6, E7]
        self.Es = [(E, [0, 0, 0]) for E in Es_raw]
        self.Es[-1] = tuple(np.add(self.Es[-1], ([0, 0, 0], [-1, 1, 0])))
    
    def setRs(self, p1: list, p2: list):
        '''
        Generates the different transition rates based on the model parameters and the given populations.

        ### Parameters
        p1: A 3-element list containing the susceptible, infected, and recovered populations. This is for the 'main' population, i.e.
        the one governed by this model's parameter set.
        p2: As above, but for the other population.
        '''
        self.Rs = [1.0,
                   1.0,
                   1.0,
                   1.0,
                   1.0,
                   1.0,
                   1.0]
        return self.Rs

    def trans(self, p: tuple[list], idx: int):
        '''
        Effects the changes in the population dictated by the simulation.

        ### Parameters
        p: A tuple of 3-element lists containing the S, I, and R values of both populations.
        idx: The index of the desired event.

        ### Returns
        p_new: The new population, in the same format as the input.
        is_hs: A bool for whether or not this was a host switch event.
        '''
        rv = np.add(p, self.Es[idx])
        p_new = tuple([list(rv[i]) for i in [0,1]])
        is_hs = bool(self.Es[idx][1][1])
        return p_new, is_hs

class SIR_base(Model):
    '''
    The standard stochastic SIR model.
    '''
    nm = 'Base'
    def setRs(self, p1: list, p2: list):
        [S, I, R] = p1
        N = sum(p1)
        self.Rs = [N*self.bd,
                    self.ir*S*I/N,
                    self.rr*I,
                    self.bd*S,
                    self.bd*I,
                    self.bd*R,
                    self.hsr*I*p2[0]/(N+sum(p2))]
        return self.Rs

class SIR_waning(SIR_base):
    '''
    Stochastic SIR model with waning immunity.
    '''
    nm = 'Waning'
    wi = -1.0
    def __init__(self, **kwargs):
        '''
        Initialises the model with the given parameters.

        ### Parameters
        bd: Birth/death rate
        ir: Infection rate
        rr: Recovery rate
        wi: Waning immunity rate
        '''
        super().__init__(**kwargs)
        self.Es.append(([1, 0, -1], [0, 0, 0]))

    def setRs(self, p1: list, p2: list):
        R = p1[-1]
        super().setRs(p1, p2)
        self.Rs.append(R*self.wi)
        return self.Rs