import numpy as np
from numpy import isscalar

def div0(a: float, 
         b: float, 
         defval: float=0):
    """ Performs ``true_divide`` but ignores the error when division by zero 
    (result is set to zero instead). """

    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        if isscalar(c):
            if not np.isfinite(c).all():
                c = defval
        else:
            c[~np.isfinite(c)] = defval  # -inf inf NaN
    return c