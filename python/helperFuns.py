import numpy as np

def deriv1Central(x):
    # first derivative using central differences
    n = x.size
    d = np.zeros((n,))
    d[0] = x[1] - x[0]
    for j,_ in enumerate(d[1:n-1],start=1):
        d[j] = x[j+1] - x[j-1]
    d[n-1] = x[n-1] - x[n-2]
    return d

def deriv1(x):
    # first derivative using adjacent differences
    n = x.size
    d = np.zeros((n,))
    d[0] = x[1] - x[0]
    for j,_ in enumerate(d[1:n],start=1):
        d[j] = x[j] - x[j-1]
    return d

def deriv2(x):
    # second derivative using 3 point central difference
    n = x.size
    d = np.zeros((n,))
    for j,_ in enumerate(d[1:n-1],start=1):
        d[j] = x[j+1] - 2*x[j] + x[j-1]
    d[0] = d[1]
    d[-1] = d[-2]
    return d