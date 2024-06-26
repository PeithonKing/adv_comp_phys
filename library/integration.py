try:
    from library.myrandom import Random
except ModuleNotFoundError:
    from myrandom import Random
 
import time
from tqdm import tqdm

from .basic_functions import legendre_derivative, find_legendre_roots

def midpoint_rule(func, a, b, n=100, verbose = False):
    """Midpoint rule for integration.

    Args:
        func (function): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        n (int, optional): Number of intervals. Defaults to 100.
        verbose (bool, optional): Print progress. Defaults to False.

    Returns:
        float: Integral of function.
    """
    h = (b-a)/n
    x = a + h/2
    s = 0
    if verbose: v = lambda x: tqdm(range(x))
    else: v = lambda x: range(x)
    for i in v(n):
        s += func(x)
        x += h
    return s*h

def trapezoidal_rule(func, a, b, n=100, verbose = False):
    """Trapezoidal rule for integration.

    Args:
        func (function): Function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        n (int, optional): Number of intervals. Defaults to 100.
        verbose (bool, optional): Print progress. Defaults to False.

    Returns:
        float: Integral of function.
    """
    h = (b-a)/n
    x = a
    s = 0
    if verbose: v = lambda x: tqdm(range(x))
    else: v = lambda x: range(x)
    for i in v(n):
        s += func(x) + func(x+h)
        x += h
    return s*h/2

def simpson_rule(func, a, b, n=100, verbose = False):
    """Simpson rule for integration.

    Args:
        func (function): Function to integrate.
        a (float): Lower bound of integration.
bound        b (float): Upper bound of integration.
        n (int, optional): Number of intervals. Defaults to 100.
        verbose (bool, optional): Print progress. Defaults to False.

    Returns:
        float: Integral of function.
    """
    h = (b-a)/n
    x = a
    s = 0
    if verbose: v = lambda x: tqdm(range(x))
    else: v = lambda x: range(x)
    for i in v(n):
        s += func(x) + 4*func(x+h/2) + func(x+h)
        x += h
    return s*h/6

def monte_carlo_integration(f, a, b, n=1e6, seed=time.time()%1, verbose = False):
    """Monte Carlo integration.

    Args:
        f (function): Function to integrate.
        a (int): Lower bound of integration.
        b (int): Upper bound of integration.
        n (int, optional): Number of points taken. Defaults to 100000.
        seed (float, optional): Seed value. Defaults to 0.1.
        verbose (bool, optional): Print progress. Defaults to False.

    Returns:
        _type_: _description_
    """
    n = int(n)
    r = Random(seed, [a, b])
    s = 0
    if verbose: v = lambda x: tqdm(range(x))
    else: v = lambda x: range(x)
    for i in v(n):
        s += f(r.LCG())
    return (b-a)/n * s  

def gaussian_quadrature(f, a, b, n):
    # if n < 1 or not isinstance(n, int):
    #     raise ValueError("n must be a positive integer")
    
    roots = find_legendre_roots(n)
    
    # print(roots)
    
    # Calculate the weights
    weights = [2 / ((1 - x**2) * legendre_derivative(x, n)**2) for x in roots]
    
    integral = 0
    for i in range(n):
        integral += weights[i] * f(((b - a) * roots[i])/2 + (b + a) / 2)
    
    return (b - a) / 2 * integral