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
        b: Birth/death rate
        i: Infection rate
        r: Recovery rate
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
        self.Es[-1] = tuple(np.add(self.Es[-1], ([0, 0, 0], [0, 1, 0])))
    
    def setRs(self, p1: list, p2: list):
        '''
        Generates the different stochastic probabilities based on the model parameters and the given population.

        ### Parameters
        pop: A 3-element list containing the susceptible, infected, and recovered populations.
        '''
        self.Rs = [1.0,
                   1.0,
                   1.0,
                   1.0,
                   1.0,
                   1.0,
                   1.0]
        self.Rs /= sum(self.Rs)
    
    def addR(self, r: float):
        self.Rs.append(r/sum(self.Rs))
        self.Rs = normalise(self.Rs)

    def trans(self, p: tuple[list], idx: int):
        '''
        Chooses an outcome for a single timestep.

        ### Parameters
        pop: A 3-element list containing the susceptible, infected, and recovered populations.
        '''
        rv = np.add(p, self.Es[idx])
        return tuple([list(rv[i]) for i in [0,1]])

class SIR_base(Model):
    '''
    The standard stochastic SIR model.
    '''
    nm = 'Base'
    def setRs(self, p1: list, p2: list):
        [S, I, R] = p1
        N = sum(p1)
        self.Rs = normalise([N*self.bd,
                            self.ir*S*I/N,
                            self.rr*I,
                            self.bd*S,
                            self.bd*I,
                            self.bd*R,
                            self.hsr*I*p2[0]])

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
        b: Birth/death rate
        i: Infection rate
        r: Recovery rate
        w: Waning immunity rate
        '''
        super().__init__(**kwargs)
        self.Es.append(([1, 0, -1], [0, 0, 0]))

    def setRs(self, p1: list, p2: list):
        R = p1[-1]
        super().setRs(p1, p2)
        self.addR(R*self.wi)

MODEL_TYPES: list[Model] = [SIR_base, SIR_waning]