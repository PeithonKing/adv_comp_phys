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

def polynomial_fit(x, y, n):
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
