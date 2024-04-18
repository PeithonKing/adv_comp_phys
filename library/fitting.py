import numpy as np
try:
    from library.matrix import Matrix
    import library.matrix as m
    from library.linear_equations import Cholesky_Decomposition, forward_propagation, backward_propagation
except ImportError:
    from matrix import Matrix
    import matrix as m
    from linear_equations import Cholesky_Decomposition, forward_propagation, backward_propagation

def linear_fit(xs, ys, sigma = None):
    if xs.shape != ys.shape:
        raise ValueError("x and y must be the same shape")
    if sigma == None:
        sigma = m.ones(xs.shape, "sigma")
    elif sigma.shape != xs.shape:
        raise ValueError("sigma must be the same shape as x and y")
    sigma2 = sigma**2
    S = (m.ones(sigma.shape)/sigma2).sum()
    Sxx = (xs.T()@(xs/sigma2)).mat[0][0]
    Syy = (ys.T()@(ys/sigma2)).mat[0][0]
    Sxy = (xs.T()@(ys/sigma2)).mat[0][0]
    Sx = (xs/sigma2).sum()
    Sy = (ys/sigma2).sum()
    Delta = S*Sxx - Sx**2
    
    a1 = (Sxx*Sy - Sx*Sxy)/Delta
    a2 = (S*Sxy - Sx*Sy)/Delta
    
    # calculating r and r**2
    r2 = Sxy**2/(Sxx*Syy)
    
    # calculating error in a1 and a2
    Delta_a1_sq = Sxx/Delta
    Delta_a2_sq = S/Delta
    
    return [a2, a1], [Delta_a2_sq**0.5, Delta_a1_sq**0.5], r2

def polynomial_fit(x, y, n):  # old function, don't use
    A = m.ones([n, n], "A", 3)
    nem = [[len(x)]+[(x**j).sum() for j in range(1, n)]]
    A[0] = nem
    for i in range(1, n):
        nem[0].pop(0)
        nem[0].append((x**(i+n-1)).sum())
        A[i] = nem
    # coefficient matrix done
	
    Y = [[sum([y.mat[k][0]*x.mat[k][0]**i for k in range(len(x))])]
         for i in range(n)]
    Y = Matrix(Y, "Y", 3)
    # intercept matrix done
	
    L = Cholesky_Decomposition(A)
    y1 = forward_propagation(L, Y)
    a = backward_propagation(L.T(), y1)

    return a

# def mypolyfit(x_, y, sigma=None, degree=1):
#     degree += 1
#     if sigma==None: sigma = np.ones(len(y))
#     x = np.zeros((x_.shape[0], degree))
    
#     # preparing x matrix: each column is 1, x, x^2, x^3, ..., x^n
#     for i in range(degree):
#         x[:, i] = np.array(x_)**i
    
#     # Theory:
#     # the Loss function is defined as: L = ((X@W - Y)/sigma)^2
#     # where W is the unknown parameters to be found
#     # our job is to minimize L by changing W
#     # at the optimum W, dL/dW = 0 (dL/dW is a vector where each element is dL/dW_i)
#     # dL/dW = 2*X.T @ ((X@W - Y)/sigma^2) = 0
#     # X.T @ (X@W - Y)/sigma^2 = 0
#     # X.T/sigma^2 @ X @ W = X.T/sigma^2 @ Y
#     # A @ W = B, where A = X.T/sigma^2 @ X and B = X.T @ Y/sigma^2
#     # now we just need to solve this linear system of equations
    
#     A = (x.T / sigma**2) @ x
#     B = x.T @ (y / sigma**2)
#     return np.linalg.solve(A, B), A

def mypolyfit(x_, y, basis=None, sigma=None, degree=1):
    degree += 1
    if sigma==None: sigma = np.ones(len(y))
    if basis == None:
        def funcs(d): return lambda x: x**d
        basis = [funcs(i) for i in range(degree)]
    else:
        assert len(basis) == degree, "basis must have the same length as degree+1"
    x = np.zeros((x_.shape[0], degree))
    
    # preparing x matrix: each column is phi_0(x), phi_1(x), phi_2(x), ..., phi_n(x)
    for i in range(degree):
        x[:, i] = basis[i](x_)
    
    A = (x.T / sigma**2) @ x
    B = x.T @ (y / sigma**2)

    return np.linalg.solve(A, B), A

def pass_all_points(x_, y):
    degree = len(y)
    x = np.zeros((x_.shape[0], degree))
    
    # preparing x matrix: each column is 1, x, x^2, x^3, ..., x^n
    for i in range(degree):
        x[:, i] = np.array(x_)**i
    
    return np.linalg.solve(x, y), x