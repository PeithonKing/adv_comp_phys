import numpy as np

# for power method

def power_method(matrix, tolerance=1e-6, max_iterations=100, seed=21):
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

def power_method_n(matrix, n, tolerance=1e-6, max_iterations=100, seed=21):
    matrix = matrix.copy()
    eigenvalues = []
    eigenvectors = []

    for _ in range(n):
        np.random.seed(seed)
        evec = np.random.rand(matrix.shape[0])
        evec = evec / np.linalg.norm(evec)

        for i in range(max_iterations):
            evec_new = matrix @ evec
            e_val = evec_new @ evec
            evec_new = evec_new / np.linalg.norm(evec_new)

            if np.linalg.norm(evec_new - evec) < tolerance:
                break

            evec = evec_new

        eigenvalues.append(e_val)
        eigenvectors.append(evec)

        # Deflate the matrix to find the next eigenvalue
        matrix -= e_val * np.outer(evec, evec)

    return np.array(eigenvalues), np.array(eigenvectors)

# for QR method

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


# for Jacobi method

def find_largest_non_diagonal(A):
    A = np.abs(A)
    n = A.shape[0]
    mask = np.eye(n, dtype=bool)
    A - np.inf * mask
    flat_upper_triangular = A[A - np.inf * mask].flatten()
    idx = np.argmax(flat_upper_triangular)
    print(idx)
    row = idx // (n - 1)
    col = idx % (n - 1) + (row + 1)
    return row, col

def rotation_matrix(N, i, j, sin_theta, cos_theta):
    R = np.eye(N)

    R[i, i] = cos_theta
    R[j, j] = cos_theta
    R[i, j] = -sin_theta
    R[j, i] = sin_theta

    return R

def jacobi_method(A, tolerance=1e-6, max_iteration=100):
    """This code finds out the eigenvalues of a matrix using Jacobi method

    Args:
        A (np.array): given matrix
        tolerance (float, optional): Tolerance. Defaults to 1e-6.
        max_iteration (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        np.array, int: Eigenvalues and number of iterations
    """
    n = A.shape[0]
    for i in range(max_iteration):
        row, col = find_largest_non_diagonal(A)
        # print(row, col, end=' ')
        # print(A[row, col])
        try: tan_2_theta = 2 * A[row, col] / (A[col, col] - A[row, row])
        except IndexError as e:
            print(A)
            print(row, col)
            raise e
        # print(tan_2_theta)
        cos_theta = 1 / np.sqrt(1 + tan_2_theta**2)
        sin_theta = cos_theta * tan_2_theta
        theta = np.arctan(sin_theta / cos_theta)
        # print(theta * 180 / np.pi)
        S = rotation_matrix(n, row, col, sin_theta, cos_theta)
        A = S.T @ A @ S
        if np.linalg.norm(np.triu(A, 1)) < tolerance:
            break
        # print(A, '\n\n')
        
    return A, i