import math

def reconstruct(original_function, theta = 0, debug = False):
    """
    Reconstructs a function f(x) = a + b cos(x + c) + d cos(x/2 + e) given as
    the `original_function` using six targeted evaluations.
    Returns the reconstruction of f.
    """

    # measure function at six chosen points
    y_0 = original_function(theta + 0)
    if debug: print(f"y_0={y_0}")

    y_pi = original_function(theta + math.pi)
    if debug: print(f"y_pi={y_pi}")

    y_32pi = original_function(theta + 3 / 2 * math.pi)
    if debug: print(f"y_3/2pi={y_32pi}")

    y_2pi = original_function(theta + 2 * math.pi)
    if debug: print(f"y_2pi={y_2pi}")

    y_3pi = original_function(theta + 3 * math.pi)
    if debug: print(f"y_3pi={y_3pi}")

    y_72pi = original_function(theta + 7 / 2 * math.pi)
    if debug: print(f"y_7/2pi={y_72pi}\n")


    # determine reconstruction constants from these measurements
    d1 = (1/4) * (y_0 + y_2pi + y_pi + y_3pi)
    if debug: print(f"d_1={d1}")

    y1_0 = (1/2) * (y_0 - y_2pi)
    y1_3pi = (1/2) * (y_3pi - y_pi)
    if y1_0 == 0 and y1_3pi == 0:
        d2 = 0 # arbitrary choice
        d3 = 0
    elif y1_0 == 0 and y1_3pi != 0:
        d2 = (1/2) * math.pi - theta / 2
        d3 = y1_3pi
    else:
        d2 = math.atan(y1_3pi / y1_0) - theta / 2
        d3 = y1_0 / math.cos(theta / 2 + d2)
    if debug: print(f"d_2={d2}\nd_3={d3}")

    y2_0 = (1/2) * (y_0 + y_2pi - 2 * d1)
    y2_32pi = (1/2) * (y_32pi + y_72pi - 2 * d1)
    if y2_0 == 0 and y2_32pi == 0:
        d4 = 0 # arbitrary choice
        d5 = 0
    elif y2_0 == 0 and y2_32pi != 0:
        d4 = (1/2) * math.pi - theta
        d5 = y2_32pi
    else:
        d4 = math.atan(y2_32pi / y2_0) - theta
        d5 = y2_0 / math.cos(theta + d4)
    if debug: print(f"d_4={d4}\nd_5={d5}\n")


    # and finally, reconstruct the cost function
    def reconstructed_function(theta):
        return d1 + d3 * math.cos(theta / 2 + d2) + d5 * math.cos(theta + d4)
    
    return reconstructed_function
