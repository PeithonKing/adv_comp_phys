try:
    from library.nonlinear_equations import solve_newton_raphson
    from library.myrandom import Random
    from library.matrix import Matrix, ones, zeros
except ImportError:
    from nonlinear_equations import solve_newton_raphson
    from myrandom import Random

import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

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
    xs, ys = [x0], [y0]
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



def heat_eq(temp:callable, Lx:float, Nx:int, Lt:float, Nt:int, needed:int):
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


def crank_nicolson_heat_eqn(u0, L, T, Nl, Nt, alpha = 1, v = 4):
    h = L / Nl
    k = T / Nt
    alpha = k*alpha / h**2

    u = np.zeros((Nl + 1, Nt + 1))
    u[:, 0] = u0

    u[0, :] = 0  # mostly this is the case
    u[Nl, :] = 0  # mostly this is the case

    # preparing the A and B matrices
    A = np.zeros((Nl-1, Nl-1))
    B = np.zeros((Nl-1, Nl-1))
    for i in range(Nl-1):
        A[i, i] = 2/v + 2 * alpha
        B[i, i] = 2/v - 2 * alpha
        if i < Nl-2:
            A[i, i+1] = -alpha
            A[i+1, i] = -alpha
            B[i, i+1] = alpha
            B[i+1, i] = alpha

    # inverting A
    A_inv = np.linalg.inv(A)

    b = np.zeros(Nl-1)
    for j in range(1,Nt+1):
        b[0]    = alpha * u[0, j-1] + alpha * u[0, j]
        b[Nl-2] = alpha * u[Nl,j-1] + alpha * u[Nl,j]
        v = B @ u[1:Nl, j-1]
        u[1:(Nl),j] = A_inv @ (v+b)
    
    return u.T, alpha

def poission_eqn(u, func, xlim=2, ylim=1):
    u = copy.deepcopy(u.T)

    N = u.shape[0]
    h = ylim / (N - 1)
    x = np.linspace(0, xlim, N)
    y = np.linspace(0, ylim, N)

    N2 = (N-1)**2
    A=np.zeros((N2, N2))
    coo = lambda i, j, N: i * N + j
    for i in range(N-1):
        for j in range(N-1):

            this = coo(i, j, N-1)
            A[this, this] = 4  # self

            if i > 0:
                A[this, coo(i-1, j, N-1)] = -1  # Left
            if i < N-2:
                A[this, coo(i+1, j, N-1)] = -1  # Right
            if j > 0:
                A[this, coo(i, j-1, N-1)] = -1  # Up
            if j < N-2:
                A[this, coo(i, j+1, N-1)] = -1  # Down

    r = np.zeros(N2)
    # vector r      
    for i in range(N-1):
        for j in range(N-1):           
            r[i+(N-1)*j] = (h**2) * func(x[i+1], y[j+1])

    # Boundary
    b_bottom_top=np.zeros(N2)
    for i in range(0,N-1):
        b_bottom_top[i]= x[i+1] #Bottom Boundary
        b_bottom_top[i+(N-1)*(N-2)] = x[i+1] * np.e# Top Boundary

    b_left_right=np.zeros(N2)
    for j in range(0, N-1):
        b_left_right[(N-1)*j] = 0 # Left Boundary
        b_left_right[N-2+(N-1)*j] = 2*np.exp(y[j+1])# Right Boundary

    b = b_left_right + b_bottom_top

    C = np.linalg.inv(A) @ (b - r)

    u[1:N, 1:N] = C.reshape((N-1, N-1))
    return u

def wave_eqn(u, delta_t=0.1, delta_x=0.1):
    h = delta_t/delta_x
    u = copy.deepcopy(u)
    Nx = u.shape[0]
    Nt = u.shape[1]
    
    u[1:Nx-1, 1] = u[1:Nx-1, 0] + (h**2)/2 * (u[0:Nx-2, 0] - 2*u[1:Nx-1, 0] + u[2:Nx, 0])
    
    for k in range(2, Nt):
        u[1:Nx-1, k] = 2 * u[1:Nx-1, k-1] - u[1:Nx-1, k-2] + (h**2) * (u[0:Nx-2, k-1] - 2*u[1:Nx-1, k-1] + u[2:Nx, k-1])
    
    return u


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    L, NL = 1, 100
    L, NT = 1, 100
    dL, dT = L/NL, L/NT

    u = np.zeros((NL, NL))

    a = 10
    b = L/2
    c = 0.1

    # gaussian pulse
    gaussian = lambda x: a * np.exp(-((x - b)**2)/(2*c**2))

    u[0, :] = 0
    u[1, :] = 0
    u[:, 0] = gaussian(np.linspace(0, L, NL))

    plt.plot(u[:, 0])
    plt.show()