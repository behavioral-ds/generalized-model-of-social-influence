import numpy as np
from functools import reduce

def P(cascade, r = -0.000068, beta = 1.0, a=None): 
    n = len(cascade)
    t = np.zeros(n,dtype = np.float64)
    f = np.zeros(n,dtype = np.float64)
    p = np.zeros((n,n),dtype = np.float64)
    norm = np.zeros(n,dtype = np.float64)
    for k, row in cascade.iterrows():
        if k == 0:
            p[0][0] = 0
            t[0] = row['time']
            if np.isnan(row['magnitude']):
                print(row)
            f[0] = 1 if row['magnitude'] == 0 else row['magnitude']
            continue
        
        t[k] = row['time']
        f[k] = (1 if row['magnitude'] == 0 else row['magnitude'])**beta
        p[:k, k] = ((r * (t[k] - t[0:k])) + np.log(f[0:k])) + (np.zeros(k,dtype = np.float64) if (a is None) else np.log(a[0:k,k])) # store the P_ji in log space
        norm[k] = reduce(np.logaddexp, p[:k, k])
        p[:k, k] = p[:k, k] - norm[k]
    return np.exp(p)

def influence(tp, alpha = None):   
    p = tp*(alpha if alpha else 1)
    n = len(p)
    m = np.zeros((n, n))
    m[0, 0] = 1
    for i in range(0, n-1):
        m[:i+1, i+1] = m[:i+1, :i+1]@p[:i+1, i+1]
        m[i+1, i+1] = (1-alpha if alpha else 1)
    influence = np.sum(m, axis = 1)
    if np.isnan(influence).any():
        raise Exception('Cascade had NaN in it')
    return influence
