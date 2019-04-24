# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from scipy.optimize import fsolve

# ------------------------------------------------------------


def lw(x):
    """
    Lambert W function (the inverse to x e^{x}), for real x >= 0.

    :param float x: The input value.
    :return: Lambert W function value for x.
    :rtype: float
    """

    def func(w, x):
        return np.log(x) - np.log(w) - w

    if x == 0:
        return 0
    if x > 2.5:
        lnx = np.log(x)
        w0 = lnx - np.log(lnx)
    elif x > 0.25:
        w0 = 0.8 * np.log(x + 1)
    else:
        w0 = x * (1.0 - x)

    return fsolve(func, w0, args=(x,))[0]


def lwexpapp(a, x):
    """
    Returns an approximation of the W(exp(a + x)) value (W is the Lambert W function)
    for small constant a and x large. Allows to calculate the value of this function
    for larger x, which is not possible directly due to an overflow in exp.

    :param float a: A parameter.
    :param float x: The function variable.
    :return: An approximation of the W(exp(a + x)) value.
    :rtype: float
    """
    return x * (1 - (math.log(x) - a) / (1 + x))

def lambertw(var):
    """
    Applies Lambert W function to a list, array or a number. For a list or array
    an numpy array is returned, for a number a single value is returned.

    :param Union[list, np.ndarray, float] var: Input.
    :return: Array of values or a single value of Lambert W function on the input.
    :rtype: Union[np.ndarray, float]
    """

    if type(var) == list:
        return np.array([lw(x) for x in var])
    else:
        return lw(var)

def lambertwexp(var):
    """
    Returns the value of a superposition of Lambert W function and exp function,
    which can be applied to larger inputs due to the use of an approximation.

    :param Union[list, np.ndarray, float] var: Input variable in the form
        of a list, numpy array or a number.
    :return: Array of values or a single value of Lambert W function
        composed with exp on the input.
    :rtype: Union[np.ndarray, float]
    """

    if type(var) == list:
        return np.array([(lw(math.exp(x)) if x < 709 else lwexpapp(0, x)) for x in var])
    else:
        return lw(math.exp(var)) if var < 709 else lwexpapp(0, var)
    return 
    

def convolve(x, y):
    """
    Calculates the convolution of an array x and a kernel array y. If the length
    of y is odd, then the zero point of the kernel is set to the center index.
    If the length is even, the zero point is set to the smaller index of the two
    in the center. The length of y must be at least 2*len(x)-1. Central 2*len(x)-1
    elements of y will be used in the calculation.

    :param np.ndarray x: An array to be convolved with a kernel.
    :param np.ndarray y: The kernel to convolve with.
    :return: A convolution of x and y with the length of x.
    :rtype: np.ndarray
    """
    
    result = np.array([0.0]*len(x))
    avg_y_index = int(math.floor(float(len(y) - 0.9) / 2))
    
    for n in range(0, len(x)):
        indices = [i for i in range(avg_y_index + n, avg_y_index + n - len(x), -1)]
        y_transformed = np.array([y[i] for i in indices])
        y_transformed = y_transformed / sum(y_transformed)
        result[n] = np.dot(x, y_transformed)
    
    return result


def get_discrete_kernel(kernel_type="gaussian", kernel_width=1, length=31):
    """
    Generates a discrete kernel of a given type, width and length.

    :param str kernel_type: Type of the kernel: "gaussian" or "uniform".
    :param float kernel_width: Standard deviation for gaussian, half the width for uniform
    :param int length: Length of the kernel.
    :return: Kernel in a form of a numpy array of length given by argument length.
        The density center of the kernel is on the central index if length is odd,
        and on the first index preceding length / 2 if length is even.
    :rtype: np.ndarray
    """
    
    kernel = np.array([0.0]*length)
    avg_index = int(math.floor(float(length - 0.9) / 2))
    
    for index in range(0, length):
        if kernel_type == "gaussian":
            kernel[index] = norm.pdf(index, avg_index, kernel_width)
    
    kernel = kernel / sum(kernel)
    
    return kernel

    
if __name__ == "__main__":

    print(convolve(np.array([1, 2, 3]), np.array([0.1, 0.3, 0.6, 1, 0.6, 0.3, 0.1])))

    # print(convolve([0, 1, 2, 3, 4, 0, 4, 3, 2, 1, 0], get_discrete_kernel()))