import math
import numpy as np

def reconstruct_rp(original_function, theta: float, value_at_theta: float, debug = False):
    # measure function at three chosen points
    y_0 = value_at_theta # cached result of original_function(theta + 0)
    y_pi = original_function(theta + math.pi)
    y_32pi = original_function(theta + (3/2) * math.pi)

    d1 = (1/2) * (y_0 + y_pi)

    y2_0 = y_0 - d1
    y2_pi = y_pi - d1
    y2_32pi = y_32pi - d1

    if y2_0 == 0 and y2_32pi == 0:
        d4 = 0 # arbitrary choice
        d5 = 0
    elif y2_0 == 0 and y2_32pi != 0:
        d4 = (1/2) * math.pi - theta
        d5 = y2_32pi
    else:
        d4 = math.atan(y2_32pi / y2_0) - theta
        # most_x = [0, math.pi, 1.5 * math.pi][np.argmax([y_0, y_pi, y_32pi])]
        d5 = y2_0 / math.cos(theta + d4)

    def reconstructed_function(theta):
        return d1 + d5 * math.cos(theta + d4)
    
    if abs(d5) > 1:
        print(f"alarm alarm! {(0.0, float(y_0 - d1))}, {(math.pi, float(y_pi - d1))}, {(1.5 * math.pi, float(y_32pi - d1))} // d4={d4}, d5={d5}")

    return reconstructed_function, {
        "d1": d1,
        "d2": 0,
        "d3": 0,
        "d4": d4,
        "d5": d5,
        "y1": lambda _: 0,
        "y2": lambda theta: reconstructed_function(theta) - d1
    }

def reconstruct_crp(original_function, theta: float, value_at_theta: float, debug = False):
    """
    Reconstructs a function f(x) = a + b cos(x + c) + d cos(x/2 + e) given as
    the `original_function` using six targeted evaluations.
    Returns the reconstruction of f.
    """

    # measure function at six chosen points
    y_0 = value_at_theta # cached result of original_function(theta + 0)
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
    def reconstructed_y1(theta):
        return d3 * math.cos(theta / 2 + d2)
    
    def reconstructed_y2(theta):
        return d5 * math.cos(theta + d4)

    def reconstructed_function(theta):
        return d1 + reconstructed_y1(theta) + reconstructed_y2(theta)
    
    return reconstructed_function, {
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "d5": d5,
        "y1": reconstructed_y1,
        "y2": reconstructed_y2
    }

def reconstruct(original_function, theta: float, value_at_theta: float, debug = False, gate = "CRP"):
    if gate == "RP":
        return reconstruct_rp(original_function, theta, value_at_theta, debug)
    elif gate == "CRP":
        return reconstruct_crp(original_function, theta, value_at_theta, debug)
    else:
        raise ValueError("unrecognized gate!", gate)
