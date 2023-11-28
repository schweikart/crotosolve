import math
from scipy.optimize import minimize
import numpy as np

def minimize_rp_reconstruction(reconstruction, constants, debug = False):
    if constants['d5'] > 0:
        x = - constants['d4']
    else:
        x = - constants['d4'] + math.pi
    return x, reconstruction(x)

def minimize_crp_reconstruction(reconstruction, constants, debug = False):
    min_y1 = 2 * math.pi - 2 * constants["d2"]
    min_y2 = np.array([1, 3]) * math.pi - constants["d4"]
    closer_min_y2 = min_y2[np.argmin(np.abs(min_y2 - min_y1))]
    x0 = (min_y1 + closer_min_y2) / 2
    
    res = minimize(reconstruction, x0, method='Nelder-Mead', tol=1e-6)

    if debug: print(res)
    return res.x[0], res.fun

def minimize_reconstruction(reconstruction, constants, debug = False, gate = "CRP"):
    if gate == "RP":
        return minimize_rp_reconstruction(reconstruction, constants, debug)
    else:
        return minimize_crp_reconstruction(reconstruction, constants, debug)
