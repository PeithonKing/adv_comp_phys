import copy
import numpy as np


def finite_element(u, func, a, b, alpha, beta):
    u = copy.deepcopy(u)

    N = u.shape[0]
    h = (b-a) / (N-1)
    x = np.linspace(a, b, N)

    A = np.zeros((N-2, N-2))
    for i in range(N-2):
        A[i, i] = 2
        if i != 0:   A[i, i-1] = -1
        if i != N-3: A[i, i+1] = -1

    r = np.zeros(N-2)
    for i in range(N-2):
        r[i] =  h**2 * func(x[i+1])
        if i == 0:   r[i] -= alpha
        if i == N-3: r[i] -= beta

    print(f"{A.shape = }, {r.shape = }")
    print(f"{A},\n{r}")

    u[1: N-1] = np.linalg.inv(A) @ r
    return -u

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    a, b = 0, 1.5
    alpha, beta = 0, 0
    N = 100
    x = np.linspace(a, b, N)
    func = lambda x: 4*x-2
    u = np.zeros(N)
    u[0] = alpha
    u[-1] = beta

    u = finite_element(u, func, a, b, alpha, beta)
    print(u)
    plt.plot(x, u, ".-", label="y(t)")
    plt.legend()
    plt.show()
