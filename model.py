import numpy as np

class Model:
    '''
    The "base" form of the model. Should not really be used directly.
    '''
    bd = -1
    ir = -1
    rr = -1
    Rs = []
    Es = []

    def __init__(self, b: float, i: float, r: float):
        '''
        Initialises the model with the given parameters.

        ### Parameters
        b: Birth/death rate
        i: Infection rate
        r: Recovery rate
        '''
        self.bd = b
        self.ir = i
        self.rr = r
    
    def setRs(self, pop: list):
        '''
        Generates the different stochastic probabilities based on the model parameters and the given population.

        ### Parameters
        pop: A 3-element list containing the susceptible, infected, and recovered populations.
        '''
        self.Rs = [1,
                   1,
                   1,
                   1,
                   1,
                   1]
        self.Rs /= sum(self.Rs)

    def trans(self, pop: list) -> list:
        '''
        Chooses an outcome for a single timestep.

        ### Parameters
        pop: A 3-element list containing the susceptible, infected, and recovered populations.
        '''
        [S, I, R] = pop
        E1 = [S+1, I, R]
        E2 = [S-1, I+1, R]
        E3 = [S, I-1, R+1]
        E4 = [S-1, I, R]
        E5 = [S, I-1, R]
        E6 = [S, I, R-1]
        self.setRs(pop)
        self.Es = [E1, E2, E3, E4, E5, E6]
        return self.Es[np.random.choice(np.arange(0,6), p=self.Rs)]

class SIR_base(Model):
    '''
    The standard stochastic SIR model.
    '''
    def setRs(self, pop: list):
        [S, I, R] = pop
        N = sum(pop)
        self.Rs = [N*self.bd,
                   self.ir*S*I/N,
                   self.rr*I,
                   self.bd*S,
                   self.bd*I,
                   self.bd*R]
        self.Rs = list(np.array(self.Rs)/sum(self.Rs))