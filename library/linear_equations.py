try:
    from matrix import Matrix, zeros, ones, randmat
    from basic_functions import mysum
except:
    from library.matrix import Matrix, zeros, ones, randmat
    from library.basic_functions import mysum

from copy import deepcopy as copy

import numpy as np

def gauss_jordan(A:Matrix, B:Matrix, verbose:bool=False):
    """Perform Gauss-Jordan elimination on the augmented matrix [A|B]

    Args:
        A (Matrix): Coefficient matrix
        B (Matrix): Intercept matrix
        verbose (bool, optional): Whether to print the steps. Defaults to False.

    Returns:
        Matrix: The solution column matrix.
    """

    A.augment(B)
    for imp in range(A.shape[0]):
        if verbose: print(f"working with row {imp}") 
        if A.mat[imp][imp] == 0:
            try: A.pivot(imp)
            except: raise ValueError("Can't do Gauss-Jordan; Try rearranging the rows.")

        A[imp] = A[imp] / A.mat[imp][imp]

        for i in range(A.shape[0]):
            if imp != i:
                A[i] -= A[imp]*A.mat[i][imp]

        print() if verbose else None  # for spacing

    ans = A#[:, -1]
    ans.name = "x"
    return ans


def forward_propagation(L, B):
    """forward propagation for cholesky decomposition

    Args:
        L (Matrix): Lower triangular matrix
        B (Matrix): Coefficient matrix

    Returns:
        Matrix: y matrix
    """
    y = zeros(B.shape, "y", B.precision)
    for i in range(len(y)):
        y[i, 0] = (B[i, 0] - L[i, :i] @ y[:i, 0])/L[i, i]
    return y


def backward_propagation(U, y):
    """backward propagation for cholesky decomposition

    Args:
        U (Matrix): Upper triangular matrix
        y (Matrix): y matrix

    Returns:
        Matrix: x matrix
    """
    x = zeros(y.shape, "x", y.precision)
    for i in range(len(x)-1, -1, -1):
        x[i, 0] = (y[i, 0] - U[i, i+1:] @ x[i+1:, 0])/U[i, i]
    return x


def LU_Decomposition(A):
    """LU decomposition

    Args:
        A (Matrix): Coefficient matrix

    Returns:
        L (Matrix): Lower triangular matrix, with 1s on the diagonal
        U (Matrix): Upper triangular matrix
    """
    n = A.shape[0]
    LU = A

    for i in range(1, n):
        for j in range(i):
            if LU.mat[j][j] == 0:
                try: A.pivot(j)
                except:raise ValueError("Can't do LU decomposition; Try rearranging the rows.")
            LU[i, j] = (LU[i, j] - LU[i, :j] @ LU[:j, j]) / LU[j, j]
        LU[i, i:] = LU[i, i:] - LU[i, :i] @ LU[:i, i:]

    # Was a single matrix till now, but while returning, I'm splitting it into L and U
    LU = LU.mat  # gives us the 2D list which has the informations about the matrix elements
    L = Matrix([[LU[i][j] if j < i else (1 if i == j else 0) for j in range(A.shape[1])] for i in range(A.shape[0])], "L", 3)
    U = Matrix([[LU[i][j] if j >= i else 0 for j in range(A.shape[1])]for i in range(A.shape[0])], "U", 3)
    return L, U


def gauss_seidel(A, B, tol=1e-6, guess=None, seed=0.1, max_iter=100):
    """Solves the system of linear equations using Gauss Siedel method.

    Args:
        A (Matrix): Coefficient matrix
        B (Matrix): Intercept matrix
        tol (float, optional): Precision. Defaults to 1e-6.
        guess (Matrix, optional): Initial guess. Defaults to random matrix with seed as seed.
        seed (float, optional): Initial seed for random matrix. Defaults to 0.1.
        max_iter (int, optional): Maximum iterations to run. Defaults to 100.

    Returns:
        x (Matrix): Solution matrix
        i (int): Number of iterations
    """
    if guess:
        if isinstance(guess, list):
            guess = Matrix(guess, "x", B.precision)
        elif not isinstance(guess, Matrix):
            raise TypeError("guess must be a list or a Matrix")
        if guess.shape != B.shape:
            raise ValueError("Guess matrix must have the same shape as B matrix.")
    x = guess if guess else randmat(B.shape, seed,  "x")
    i = 0
    while True:
        x_old = copy(x)
        for i in range(len(x)):
            x[i,0] = (B[i,0] - A[i,:i]@x[:i,0] - A[i, i+1:]@x_old[i+1:,0])/A[i, i]
        i += 1
        if abs(x-x_old).sum()/len(x) < tol or i > max_iter:
            break
    x.name = "x"
    return x, i


def jacobi(A, B, tol=1e-6, guess=None, seed=0.1, max_iter=100):
    """Solves the system of linear equations using Gauss Siedel method.

    Args:
        A (Matrix): Coefficient matrix
        B (Matrix): Intercept matrix
        tol (float, optional): Precision. Defaults to 1e-6.
        guess (Matrix, optional): Initial guess. Defaults to random matrix with seed as seed.
        seed (float, optional): Initial seed for random matrix. Defaults to 0.1.
        max_iter (int, optional): Maximum iterations to run. Defaults to 100.

    Returns:
        x (Matrix): Solution matrix
        i (int): Number of iterations
    """
    D = zeros(B.shape, "C", B.precision)
    for i in range(len(A)):
        D[i, 0] = A[i, i]
        A[i, i] = 0

    if guess:
        if isinstance(guess, list):
            guess = Matrix(guess, "x", B.precision)
        elif not isinstance(guess, Matrix):
            raise TypeError("guess must be a list or a Matrix")
        if guess.shape != B.shape:
            raise ValueError("Guess matrix must have the same shape as B matrix.")
    x = guess if guess else randmat(B.shape, seed,  "x")
    i = 0
    while True:
        x_old = x
        x = (B - A@x)/D
        i += 1
        if abs(x-x_old).sum() < tol or i > max_iter:
            break
    x.name = "x"
    return x, i


def Cholesky_Decomposition(A):
    """Returns the Cholesky decomposition of the matrix.

    Returns:
        L (Matrix): Lower Triangular matrix L such that A = L@Lᵀ.
    """
    if not A.is_symmetric():
        raise ValueError(
            "Matrix is not symmetric: cannot perform Cholesky decomposition.")

    L = zeros(A.shape, "L", A.precision)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i == j:
                L[i, i] = (A[i, i] - L[i,:i] @ L[i, :i].T()).mat[0][0]**0.5
            elif i > j:
                L[i, j] = (A[i, j] - L[i,:i] @ L[j, :i].T()) / L.mat[j][j]
            else: break  # to save some small amount of time (and complexity)
    return L


def make_diag_dominant(A, B):
    """Makes the matrix diagonally dominant.

    Args:
        A (Matrix): Matrix to be made diagonally dominant

    Returns:
        Matrix: Diagonally dominant matrix
    """
    for i in range(len(A)):
        # ind = index of max element of ith row
        ind = max(enumerate(A[i].mat[0]), key=lambda x: x[1])[0]
        if ind<i:
            raise ValueError("Matrix cannot be made diagonally dominant.")
        A.swap_rows(i, ind, False)
        B.swap_rows(i, ind, False)
    return A, B

def conjugate_gradient(A, b, x0, tol=1e-6, max_iter=None):
    """
    Solve the linear system Ax = b using the conjugate gradient method.

    Parameters:
    A : numpy.ndarray
        Coefficient matrix of shape (n, n).
    b : numpy.ndarray
        Right-hand side vector of shape (n,).
    x0 : numpy.ndarray
        Initial guess for the solution vector of shape (n,).
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is None (no limit).

    Returns:
    x : numpy.ndarray
        Solution vector.
    iter_count : int
        Number of iterations performed.
    """
    
    A = copy(A)

    n = len(b)
    x = x0.copy()
    r = b - np.dot(A, x)
    p = r.copy()
    iter_count = 0
    
    residues = []

    while True:
        iter_count += 1
        Ap = np.dot(A, p)
        alpha = np.dot(r.T, r) / np.dot(p.T, Ap)
        x += alpha * p
        r_next = r - alpha * Ap
        residues.append(np.linalg.norm(r_next))
        if np.linalg.norm(r_next) < tol:
            break
        beta = np.dot(r_next.T, r_next) / np.dot(r.T, r)
        p = r_next + beta * p
        r = r_next

        if max_iter is not None and iter_count >= max_iter:
            break

    return x, iter_count, residues



if __name__ == "__main__":
    A = Matrix(
        [
            [1, -1, 4, 0, 2, 9],
            [0, 5, -2, 7, 8, 4],
            [1, 0, 5, 7, 3, -2],
            [6, -1, 2, 3, 0, 8],
            [-4, 2, 0, 5, -5, 3],
            [0, 7, -1, 5, 4, -2],
        ], "A", 3
    )
    L, U = LU_Decomposition(A)
    print(L)