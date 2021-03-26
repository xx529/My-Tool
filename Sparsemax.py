import numpy as np

def sparsemax(vector):
    
    # sort vector
    temp = -np.sort(-vector) 

    # compute kz
    cumsum = 0
    for i in range(len(temp)):
        cumsum += temp[i]
        if 1 + (1+i) * temp[i] > cumsum:
            kz = i + 1

    # compute tor_z
    tor_z = (temp[:kz].sum() - 1) / kz

    # output
    result = np.maximum(vector-tor_z, 0)
  
    return result
