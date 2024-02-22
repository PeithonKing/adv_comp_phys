import warnings
from math import log10

from .myrandom import Random
from .matrix import truncate_p, Matrix

def P(x, coeff):  # Polynomial
    return sum([c*x**(len(coeff)-i-1) for i, c in enumerate(coeff)])

def dP(coeff):  # derivative of polynomial
    coeff = coeff[::-1]
    dcoeff = [coeff[i] * i for i in range(len(coeff))]
    return dcoeff[-1:0:-1]

def get_brackets(f, guess = None, alpha=0.1, seed=0.2):
    # print(guess)
    r = Random(seed, [-5, 5])
    if not guess: guess = [r.LCG(), r.LCG()]
    guess.sort(reverse=True)
    a, b = guess
    # print(f"Finding brackets...{a}, {b}")
    if f(a)*f(b) < 0:
        return [a, b]
    if abs(f(a)) > abs(f(b)):
        b += alpha*(b-a)
    elif abs(f(b)) > abs(f(a)):
        a += alpha*(b-a)
    else:  # i.e.: if f(a) == f(b)
        r = Random(r.LCG(), [-5, 5])  # a+b is a random number
        guess = [r.LCG(), r.LCG()]
    return get_brackets(f, [a, b], seed = r.LCG())

def solve_bisection(f, guess = None, epsilon = 1e-4, delta=1e-4, rec_depth=0, verbose=False):
    a, b = get_brackets(f, guess)
    c = (a+b)/2
    if f(a)*f(c) > 0:
        a = c
    elif f(b)*f(c) > 0:
        b = c
    else:
        raise ValueError("No root in interval")
    places = int(-log10(delta))
    if verbose:
        x_p = [a, b][abs(f(a))>abs(f(b))]
        print(f"step={rec_depth+1}\t  x={truncate_p(x_p, places, str)}\tf(x)={truncate_p(f(x_p), places, str)}")
    return truncate_p([a, b][abs(f(a))>abs(f(b))], places, str) if (abs(a-b) < epsilon or abs(f(a))<delta or abs(f(b))<delta) else solve_bisection(f, [a, b], epsilon, delta, rec_depth+1, verbose)

def solve_regula_falsi(f, guess = None, delta=1e-4, rec_depth = 0, verbose = False):
    # print(f"rf: step={rec_depth+1}")
    a, b = get_brackets(f, guess)
    c = a + (b-a)*abs(f(a))/(abs(f(a))+abs(f(b)))
    if f(a)*f(c) > 0: a = c
    else: b = c
    
    res = [a, b][abs(f(a))>abs(f(b))]
    places = int(-log10(delta))
    if verbose:
        x_p = [a, b][abs(f(a))>abs(f(b))]
        print(f"step={rec_depth+1}\t  x={truncate_p(x_p, places, str)}\tf(x)={truncate_p(f(x_p), places, str)}")
    ret = solve_regula_falsi(f, [a, b], delta, rec_depth+1, verbose) if (abs(f(a))>delta and abs(f(b))>delta)     else res
    return ret if rec_depth else truncate_p(ret, places, str)

def differentiate(f, x, epsilon = 1e-6):  # Numerical differentiation
    return (f(x+epsilon)-f(x))/epsilon

def solve_newton_raphson(f, f_d = None, guess = None, delta=1e-4, rec_depth = 0, verbose=False):
    # print(f"nr: step={rec_depth+1}")
    if f_d==None:
        warnings.warn("No derivative provided, using numerical differentiation")
        f_d = lambda x: differentiate(f, x)
    if guess == None:
        r = Random(0.2, [-5, 5])
        guess = r.LCG()
    guess = guess - f(guess)/f_d(guess)
    places = int(-log10(delta))
    if verbose:
        print(f"step={rec_depth+1}\t  x={guess}\tf(x)={f(guess)}")
    return solve_newton_raphson(f, f_d, guess, delta, rec_depth+1, verbose) if (abs(f(guess))>delta) else guess

def laguerre(coeff, b = None, epsilon = 1e-6):
    if b == None:
        r = Random(0.1, [-5, 5])
        b = r.LCG()
    for n in range(1000):
        # print(".", end = "")
        if abs(P(b, coeff))<epsilon: return b
        
        G = P(b, dP(coeff))/P(b, coeff)
        H = G**2 - P(b, dP(dP(coeff)))/P(b, coeff)
        n = len(coeff)
        D = ((n-1)*(n*H-G*G))**0.5
        
        if abs(G+D) > abs(G-D):
            a = n/(G+D)
        else:
            a = n/(G-D)

        if abs(a) < epsilon: return b
        b = b - a

def deflation(coeff, a):  # coeff = [-1, 3, 0, -4]
    l = [coeff[0]]
    i = 1
    for i in range(len(coeff)-1):
        l.append(coeff[i+1] + a*l[-1])
    return l[:-1]

def laguerre_solve(coeff, epsilon = 1e-6):
    roots = []
    for i in range(len(coeff)-1):
        roots.append(laguerre(coeff, 0, epsilon))
        coeff = deflation(coeff, roots[-1])
        # print()
    return Matrix([[truncate_p(root, int(-log10(epsilon)), str)] for root in roots], "a", int(-log10(epsilon)))


def fixed_point_iteration(phi, x0, max_it=100, tolerance=1e-4):
    """
    Fixed-point iteration method for solving equations of the form x = phi(x).

    Parameters:
    - phi (function): Function defining the fixed-point iteration.
    - x0 (float): Initial guess.
    - max_it (int, optional): Maximum number of iterations. Defaults to 100.
    - tolerance (float, optional): Convergence tolerance. Defaults to 1e-4.

    Returns:
    - float: Approximate solution of the equation.

    Raises:
    - AssertionError: If the derivative of the function at the fixed point is greater than or equal to 1.
    """

    assert abs(differentiate(phi, x0)) < 1, "The derivative of the function at the fixed point is greater than or equal to 1. The fixed-point iteration may not converge."

    for i in range(1, max_it + 1):
        x = phi(x0)
        # print(f"Iteration {i}: x = {x}")
        if abs(x - x0) < tolerance:
            return x
        x0 = x

    return x0



