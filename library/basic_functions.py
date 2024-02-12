import math
import numpy as np
from .nonlinear_equations import solve_newton_raphson

def mysum(a):
    """iteratively adds all elements of a list

    Args:
        a (iterable): list of numbers
    """
    try:
        total = a[0]
        for i in a[1:]:
            total += i
        return total
    except:
        return 0

def product(a):
    """iteratively multiplies all elements of a list

    Args:
        a (iterable): list of numbers
    """
    if len(a) == 0:
        raise Exception("Empty list, cannot calculate product")
    total = 1
    for i in a:
        total *= i
    return total

def differentiate(f, x, epsilon = 1e-6):  # Numerical differentiation
    return (f(x+epsilon)-f(x))/epsilon


def truncate_to_decimal_places(number, decimal_places):
    """
    Truncate a number to a specified number of decimal places.

    Args:
    - number (float): Number to truncate.
    - decimal_places (int): Number of decimal places to truncate to.

    Returns:
    - float: Truncated number.
    """
    scale = 10 ** decimal_places
    truncated_number = math.floor(number * scale) / scale
    return truncated_number


def legendre_pol(x, n):
    """
    Legendre polynomial of degree n evaluated at x.

    Args:
    - x (float or array-like): Value or values at which to evaluate the Legendre polynomial.
    - n (int): Degree of the Legendre polynomial.

    Returns:
    - float or ndarray: Legendre polynomial value(s) at x.
    """
    # Ensure n is a non-negative integer
    if n < 0 or not isinstance(n, int):
        raise ValueError("n must be a non-negative integer")

    # Ensure x is in the range [-1, 1]
    if np.any(x < -1) or np.any(x > 1):
        raise ValueError("x must be in the range [-1, 1]")

    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        return ((2 * n - 1) * x * legendre_pol(x, n-1) - (n-1) * legendre_pol(x, n-2)) / n


def legendre_derivative(x, n, h=1e-6):
    """
    Numerical derivative of Legendre polynomial of degree n at x.

    Args:
    - x (float): Value at which to evaluate the derivative.
    - n (int): Degree of the Legendre polynomial.
    - h (float, optional): Step size for numerical differentiation. Defaults to 1e-6.

    Returns:
    - float: Numerical derivative of the Legendre polynomial at x.
    """
    # Ensure x is not 1.0 or -1.0
    # if x == 1.0 or x == -1.0:
    #     raise ValueError("x cannot be 1 or -1")

    return (legendre_pol(x + h/2, n) - legendre_pol(x - h/2, n)) / h

# from tqdm import trange

def find_legendre_roots(n):
    roots = []
    for i in range(1, n + 1):
        x0 = np.cos((2*i - 1) * np.pi / (2 * n))  # Initial guess using Chebyshev nodes
        root = solve_newton_raphson(lambda x: legendre_pol(x, n), lambda x: legendre_derivative(x, n), x0)
        if root is not None:
            roots.append(root)
    return roots