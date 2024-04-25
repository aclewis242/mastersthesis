from func_lib import *
import numpy as np

class Model:
    '''
    The "base" form of the model. Should not really be used directly.
    '''
    bd = -1.0
    ir = -1.0
    rr = -1.0
    Rs = []
    Es = []

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
        self.Es = [E1, E2, E3, E4, E5, E6]
    
    def setRs(self, pop: list):
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
                   1.0]
        self.Rs /= sum(self.Rs)
    
    def addR(self, r: float):
        self.Rs.append(r/sum(self.Rs))
        self.Rs = normalise(self.Rs)

    def trans(self, pop: list):
        '''
        Chooses an outcome for a single timestep.

        ### Parameters
        pop: A 3-element list containing the susceptible, infected, and recovered populations.
        '''
        self.setRs(pop)
        return list(np.add(pop, self.Es[np.random.choice(np.arange(0,len(self.Rs)), p=self.Rs)]))

class SIR_base(Model):
    '''
    The standard stochastic SIR model.
    '''
    def setRs(self, pop: list):
        [S, I, R] = pop
        N = sum(pop)
        self.Rs = normalise([N*self.bd,
                            self.ir*S*I/N,
                            self.rr*I,
                            self.bd*S,
                            self.bd*I,
                            self.bd*R])

class SIR_waning(SIR_base):
    '''
    Stochastic SIR model with waning immunity.
    '''
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
        self.Es.append([1, 0, -1])

    def setRs(self, pop: list):
        R = pop[-1]
        super().setRs(pop)
        self.addR(R*self.wi)

MODEL_TYPES: list[Model] = [SIR_base, SIR_waning]