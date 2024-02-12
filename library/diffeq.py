try:
    from library.nonlinear_equations import solve_newton_raphson
    from library.myrandom import Random
    from library.matrix import Matrix, ones, zeros
except ImportError:
    from nonlinear_equations import solve_newton_raphson
    from myrandom import Random

import matplotlib.pyplot as plt
from tqdm import tqdm

def forward_euler(f, x, x0, y0, h=0.01):
    """Forward Euler method for solving ODEs.

    Args:
        f (function): Function to be integrated.
        x (int): x value to be evaluated.
        x0 (float): Initial x value.
        y0 (float): Initial y value.
        h (float): Step size.

    Returns:
        float: Approximate solution to the ODE.
    """
    xs, ys = [], []
    while x0 < x:
        y0 += h * f(x0, y0)
        x0 += h
        xs.append(x0)
        ys.append(y0)
    return xs, ys, y0


def backward_euler(f, x, x0, y0, h=0.01):
    """Backward Euler method for solving ODEs.

    Args:
        f (function): Function to be integrated.
        x (int): x value to be evaluated.
        x0 (float): Initial x value.
        y0 (float): Initial y value.
        h (float): Step size.

    Returns:
        float: Approximate solution to the ODE.
    """
    xs, ys = [], []
    while x0 < x:
        def fn(y): return y0 + h * f(x0 + h, y) - y
        ynp1 = solve_newton_raphson(fn)
        x0 += h
        y0 += h * f(x0, float(ynp1))
        xs.append(x0)
        ys.append(y0)
    return xs, ys, y0


def predictor_corrector(f, x, x0, y0, h=0.01):
    """Predictor-Corrector method for solving ODEs.

    Args:
        f (function): Function to be integrated.
        x (int): x value to be evaluated.
        x0 (float): Initial x value.
        y0 (float): Initial y value.
        h (float): Step size.

    Returns:
        float: Approximate solution to the ODE.
    """
    xs, ys = [], []
    while x0 < x:
        y1 = y0 + h * f(x0, y0)
        y2 = y0 + h * f(x0 + h, y1)
        y0 = y0 + h * (f(x0, y0) + f(x0 + h, y2)) / 2
        x0 += h
        xs.append(x0)
        ys.append(y0)
    return xs, ys, y0


def rk2(f, x, x0, y0, h=0.01):
    """Second-order Runge-Kutta method for solving ODEs.

    Args:
        f (function): Function to be integrated.
        x (int): x value to be evaluated.
        x0 (float): Initial x value.
        y0 (float): Initial y value.
        h (float): Step size.

    Returns:
        float: Approximate solution to the ODE.
    """
    xs, ys = [], []
    while x0 < x:
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + h, y0 + k1)
        y0 += (k1 + k2) / 2
        x0 += h
        xs.append(x0)
        ys.append(y0)
    return xs, ys, y0


def rk4(f, x, x0, y0, h=0.01):
    """Fourth-order Runge-Kutta method for solving ODEs.

    Args:
        f (function): Function to be integrated.
        x (int): x value to be evaluated.
        x0 (float): Initial x value.
        y0 (float): Initial y value.
        h (float): Step size.

    Returns:
        float: Approximate solution to the ODE.
    """
    xs, ys = [], []
    while x0 < x:
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + h / 2, y0 + k1 / 2)
        k3 = h * f(x0 + h / 2, y0 + k2 / 2)
        k4 = h * f(x0 + h, y0 + k3)
        y0 += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x0 += h
        xs.append(x0)
        ys.append(y0)
    return xs, ys, y0

def c_ode(funcs, var, x_get, h=0.01):
    """Solves a coupled ODE system of n equations.

    Args:
        funcs (list[callable, ...]): Functions to be integrated.
        var (iterable[float, ...]): Initial values of the variables.
        x_get (float): Final x value.
        h (float, optional): Step size.. Defaults to 0.01.

    Raises:
        ValueError: If the number of functions is not 1 less than the number of variables.

    Returns:
        evol (2D list of shape (len(var), i)): Approximate solutions to the ODE system. [i is the number of iterations taken]
    """
    if len(funcs) + 1 != len(var):
        raise ValueError(f"Number of functions must be one less than number of variables. FYI: ({len(funcs) = }) + 1 = {len(funcs)+1} != ({len(var) = })")
    evol = [[var_i] for var_i in var]
    k = [[None for _ in range(len(funcs))] for __ in range(4)]
    count = 0
    while var[0] <= x_get:
        count += 1
        k[0] = [h * funcs[i](*var) for i in range(len(funcs))]
        k[1] = [h * funcs[i](var[0] + h/2, *(var[j] + k[0][j-1]/2 for j in range(1, len(var)))) for i in range(len(funcs))]
        k[2] = [h * funcs[i](var[0] + h/2, *(var[j] + k[1][j-1]/2 for j in range(1, len(var)))) for i in range(len(funcs))]
        k[3] = [h * funcs[i](var[0] + h, *(var[j] + k[2][j-1] for j in range(1, len(var)))) for i in range(len(funcs))]
        var[0] += h
        evol[0].append(var[0])
        for i in range(1, len(var)):
            var[i] += (1/6)*(k[0][i-1]+2*k[1][i-1]+2*k[2][i-1]+k[3][i-1])
            evol[i].append(var[i])
    return evol

def lag_interpol(zeta_h, zeta_l, yh, yl, y):
    return zeta_l + (zeta_h - zeta_l) * (y - yl)/(yh - yl)

def shooting_method(
    f1: callable,
    f2: callable,
    a: float,
    alpha: float,
    b: float,
    beta: float,
    h: float = 0.01,
    zeta_l: float = None,
    change: float = 1,
    seed:float =0.56,
    epsilon: float = 1e-6
):
    """Solves a coupled ODE system of two equations using the shooting method.

    Args:
        f1 (callable): First function to be integrated.
        f2 (callable): Second function to be integrated.
        a (float): Initial x value.
        alpha (float): Initial y value.
        b (float): Initial z value.
        beta (float): Initial z value.
        h (float, optional): Step Size. Defaults to 0.01.
        zeta_l (float, optional): Guess value of zeta. Defaults to None.
        change (float, optional): Amount to change. Defaults to 1.
        seed (float, optional): Seed if zeta_l not provided. Defaults to 0.56.
        epsilon (float, optional): Tollerance. Defaults to 1e-6.

    Returns:
        X, Y, Z: Approximate solutions to the ODE system.
    """
    if zeta_l is None:
        random = Random(seed)
        zeta_l = random.LCG()

    x, y, z = c_ode([f1, f2], [a, alpha, zeta_l], b, h)
    yh = y[-1]
    if abs(yh - beta) < epsilon: return x, y, z
    sign0 = yh > beta  # True if yn > beta, False if yn < beta

    diff0 = abs(yh - beta)

    while True:
        zeta_h = zeta_l
        zeta_l -= change
        # print(f"{zeta_l = }")
        x, y, z = c_ode([f1, f2], [a, alpha, zeta_l], b, h)
        yl = y[-1]
        if abs(yl - beta) < epsilon:
            return x, y, z
        if diff0 < abs(yl - beta): change = -change
        if (yl > beta) != sign0:
            break
        sign0 = yh > beta

    zeta_hope = lag_interpol(zeta_h, zeta_l, yh, yl, beta)

    x, y, z = c_ode([f1, f2], [a, alpha, zeta_hope], b, h)
    yh = y[-1]
    # return x, y, z
    if abs(yh - beta) < epsilon:
        return x, y, z
    else: 
        return shooting_method(f1, f2, a, alpha, b, beta, h, zeta_hope, change/10, epsilon = epsilon)



def heat_eq2(temp:callable, Lx:float, Nx:int, Lt:float, Nt:int, needed:int):
    """Solves the heat equation in 1D.

    Args:
        temp (callable): Initial temperature distribution.
        Lx (float): Length of the rod.
        Nx (int): Number of grid points in x.
        Lt (float): Time to solve for.
        Nt (int): Number of time steps.
        needed (int): Upto the number of time steps to actually calculate.
    
    Returns:
        A: Approximate solution to the heat equation.
           Where each row is a time step, and each column
           is a point on the rod.
    """
    hx = Lx/Nx
    ht = Lt/Nt
    alpha = ht/(hx**2)
    print(f"{alpha=}")

    A = zeros((needed, Nx)).mat
    for i in range(Nx): A[0][i] = temp(Nx, i)
    for t in tqdm(range(1, needed)):
        for x in range(Nx):
            if x == 0:       A[t][x] = 0                 + (1 - 2*alpha)*A[t-1][x] + alpha*A[t-1][x+1]
            elif x == Nx-1:  A[t][x] = alpha*A[t-1][x-1] + (1 - 2*alpha)*A[t-1][x] + 0
            else:            A[t][x] = alpha*A[t-1][x-1] + (1 - 2*alpha)*A[t-1][x] + alpha*A[t-1][x+1]
    
    return A
