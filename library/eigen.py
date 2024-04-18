import numpy as np


# def get_eigen(A:Matrix, epsilon:float = 1e-6, seed:float = 0.1):
#     """Get the principal eigenvalue and eigenvectors of a matrix using the power method.

#     Args:
#         A (Matrix): Matrix to be analyzed.
#         epsilon (float, optional): Tolerance. Defaults to 1e-6.
#         seed (float, optional): Initial guess. Defaults to 0.1.

#     Returns:
#         eigenvalue, eigenvector, iterations: Principal eigenvalue and eigenvector, and number of iterations.
#     """
#     precision = int(-math.log10(epsilon))
#     x0 = randmat((len(A), 1), name="x0", precision=precision, seed=seed)

#     z = x0
#     i = 0
#     lambda_old = 0
#     while True:
#         y = z
#         z = A@y
#         lambda_new = ((z.T()@x0)/(y.T()@x0)).mat[0][0]
#         i += 1
#         # print(f"{lambda_new = }")
#         if abs(lambda_old-lambda_new)<=epsilon:
#             break
#         lambda_old = lambda_new
    
#     y = y.normalise()
    
#     y.name = "e-vector"
#     return truncate_p(lambda_new, precision, str), y, i

def power_method(matrix, tolerance=1e-6, max_iterations=100, seed=10):
    """This code finds out the maximum eigenvalues and eigenvectors of a matrix

    Args:
        matrix (array): given matrix
        tolerance (float, optional): the tolerance. Defaults to 1e-6.
        max_iterations (int, optional): the maximum no of iterations. Defaults to 1000.

    Returns:
        eigvalue, eigvector, iterations: Maximum eigenvalue, eigenvector, and number of iterations taken
    """
    matrix = matrix.copy()
    n = matrix.shape[0]
    np.random.seed(seed)
    evec = np.random.rand(n)
    evec = evec / np.linalg.norm(evec)

    for i in range(max_iterations):
        evec_new = matrix @ evec
        e_val = evec_new @ evec
        evec_new = evec_new / np.linalg.norm(evec_new)

        if np.linalg.norm(evec_new - evec) < tolerance:
            break

        evec = evec_new

    return e_val, evec_new, i

def _QR_factorize(matrix):  # internal function not to be used outside
    Q_matrix = np.zeros(matrix.shape)
    R_matrix = np.zeros(matrix.shape)

    for i in range(matrix.shape[1]):
        minus = (matrix[:, i] @ Q_matrix[:, :i]) * Q_matrix[:, :i]
        v_i = matrix[:, i] - minus.sum(axis=1)
        Q_matrix[:, i] = v_i / np.linalg.norm(v_i)
        R_matrix[:i+1, i] = matrix[:, i] @ Q_matrix[:, :i+1]
    
    return Q_matrix, R_matrix


def qr_method(A, tolerance = 1e-6, max_iteration=100):
    """This code finds out the eigenvalues of a matrix using QR decomposition

    Args:
        A (np.array): given matrix
        tolerance (float, optional): Tolerance. Defaults to 1e-6.
        max_iteration (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        np.array, int: Eigenvalues and number of iterations
    """
    for i in range(max_iteration):
        Q, R = _QR_factorize(A)
        new_A = R @ Q
        if np.linalg.norm(A - new_A) < tolerance:
            break
        A = new_A

    return np.diag(A), i