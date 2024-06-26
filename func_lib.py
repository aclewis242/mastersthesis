import numpy as np

def normalise(l: np.ndarray[float]):
    '''
    Normalises the given list such that its sum is 1.
    '''
    return l/sum(l)

def fn(f_n: str):
    '''
    Converts the given string into a proper file name.
    '''
    return f_n.lower().replace(' ', '_')