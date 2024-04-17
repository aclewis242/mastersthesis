import numpy as np

class Model:
    mu = -1
    beta = -1
    gamma = -1
    Rs = []
    Es = []

    def __init__(self, m, b, g):
        self.mu = m
        self.beta = b
        self.gamma = g
    
    def setRs(self, pop):
        self.Rs = [1,
                   1,
                   1,
                   1,
                   1,
                   1]
        self.Rs /= sum(self.Rs)

    def genPs(self, pop):
        [X, Y, Z] = pop
        E1 = [X+1, Y, Z]
        E2 = [X-1, Y+1, Z]
        E3 = [X, Y-1, Z+1]
        E4 = [X-1, Y, Z]
        E5 = [X, Y-1, Z]
        E6 = [X, Y, Z-1]
        self.setRs(self, pop)
        self.Es = [E1, E2, E3, E4, E5, E6]

    def trans(self, pop):
        if len(self.Rs) == 0 or len(self.Es == 0):
            self.genPs(pop)
        return self.Es[np.random.choice(np.arange(0,5), p=self.Rs)]

class SIR_base(Model):
    def setRs(self, pop):
        [X, Y, Z] = pop
        N = sum(pop)
        self.Rs = [N*self.mu,
                   self.beta*X*Y/N,
                   self.gamma*Y,
                   self.mu*X,
                   self.mu*Y,
                   self.mu*Z]