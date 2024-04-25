import numpy as np

def normalise(l: list[float]) -> list[float]:
    return list(np.array(l)/sum(l))