from func_lib import *
from population import *
from allele import *
from operator import add
import random

class SIR:
    '''
    The stochastic SIR model class.
    '''
    pn = ''
    pn_full = ''
    sn = ''
    genotype = ''
    alleles = []
    bd = -1.0
    ir = -1.0
    rr = -1.0
    wi = -1.0
    itr = {}
    mr = 0.0
    dt = 0.0
    Rs = []
    Es = []
    num_Es = -1
    does_mutate = False
    def __init__(self, p0: population, **kwargs):
        '''
        Initialises the model with the given parameters.

        ### Parameters
        p0: Initial population, as a 3-element list (S, I, R)
        pn: The name (short) of the population this model belongs to
        pn_full: The full name of the population this model belongs to
        sn: The name of the strain this model belongs to
        bd: Birth/death rate
        ir: Infection rate
        rr: Recovery rate
        wi: Waning immunity rate
        itr: Interspecific transmission rates from this model to others (dict)
        mr: Mutation rate
        '''
        self.pop = p0
        self.__dict__.update(kwargs)
        if 'itr' in kwargs.keys(): self.itr = dict(kwargs['itr'])
        self.pop.pn = self.pn
        self.pop.addStrain(self.sn)
        E1 = [1, 0, 0]  # Birth
        E2 = [-1, 1, 0] # Infection
        E3 = [0, -1, 1] # Recovery
        E4 = [-1, 0, 0] # Death of susceptible
        E5 = [0, -1, 0] # Death of infected
        E6 = [0, 0, -1] # Death of recovered
        E7 = [1, 0, -1] # Waning immunity
        E8 = [0, -1, 0] # Mutation
        self.Es = [E1, E2, E3, E4, E5, E6, E7, E8]
        self.num_Es = len(self.Es)
        self.setRs()

    def setRs(self):
        '''
        Generates the different transition rates based on the model's parameters and population.
        '''
        [S, I, R] = self.pop.getPop(self.sn)
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
                    self.wi*R,
                    self.mr*I] + [self.itr[p2]*I*p2.sus/(N+p2.tot_pop) for p2 in self.itr]
        return self.Rs

    def trans(self, idx: int, rpt: int=1):
        '''
        Effects the changes in the population dictated by the simulation.

        ### Parameters
        idx: The index of the desired event, corresponding with the order of Rs.
        rpt: The number of times to repeat said event.
        '''
        pop = self.pop
        sn = self.sn
        if idx == self.num_Es-1:
            sn = random.choice(list(pop.inf.keys()))
            idx = 1
        if idx >= self.num_Es:
            pop = list(self.itr.keys())[idx-self.num_Es]
            idx = 1
        pop.addPop(list(map(lambda x: float(rpt)*x, self.Es[idx])), sn)
        return self.pop.getPop(self.sn)
    
    def newStrain(self, nsn='new'):
        '''
        Generates a copy of this model with the given strain name.
        '''
        new_mdl = SIR(self.pop, sn=nsn, pn=self.pn)
        new_mdl.__dict__.update(self.__dict__)
        new_mdl.sn = nsn
        new_mdl.itr = dict(new_mdl.itr)
        return new_mdl
    
    def mutate(self, param: str, fac: float):
        '''
        Effects the given parameter change.

        ### Parameters
        param: The parameter of the model to change.
        fac: The numerical factor to change it by. This value is used directly.
        '''
        if type(self.__dict__[param]) is dict:
            for k in self.__dict__[param]: self.__dict__[param][k] *= fac
        else: self.__dict__[param] *= fac
    
    def mutateMult(self, params: list[str], fac: float):
        '''
        Effects the given parameter changes.

        ### Parameters
        params: The parameters of the model to change.
        fac: The numerical factor to change them by (all the same). This value is used directly.
        '''
        for p in params: self.mutate(p, fac)
    
    def updateGenotype(self, g: str, alleles: list[allele]):
        '''
        Generates a new model based on the given genotype.

        ### Parameters
        g: The genotype, as a string of characters corresponding to alleles.
        alleles: The list of all possible alleles, as allele objects.
        '''
        new_model = self.newStrain(g)
        new_model.genotype = g
        g_real = [min(l) for l in g.split('.')]
        for a in alleles:
            if a.char in g_real:
                if a.fav_pop == new_model.pn: new_model.mutate(a.param, 1+a.fac)
                if a.unf_pop == new_model.pn: new_model.mutate(a.param, 1-a.fac)
        return new_model
    
    def __str__(self):
        return f'population {self.pn}, strain {self.sn}'
    
    def __repr__(self):
        return self.__str__()