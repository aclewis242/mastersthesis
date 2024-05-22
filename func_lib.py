import numpy as np

def normalise(l: list[float]) -> list[float]:
    '''
    !! DEPRECATED !!
    '''
    return list(np.array(l)/sum(l))