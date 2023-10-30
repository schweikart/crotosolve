import math
from scipy.optimize import minimize

def minimize_reconstruction(reconstruction, x0 = math.pi, debug = False):
    x0 = math.pi
    res = minimize(reconstruction, x0, method='Nelder-Mead', tol=1e-6)

    if debug: print(res)
    return res.x[0], res.fun
