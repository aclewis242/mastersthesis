import numpy as np

def normalise(l: list[float]) -> list[float]: # deprecated
    return list(np.array(l)/sum(l))