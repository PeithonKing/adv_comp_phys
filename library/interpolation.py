from library.basic_functions import product

def interpolate(p, x, y, N=None):
    if N is None:
        N = len(x)
    if len(x) != len(y) and min([len(x), len(y)])<N:
        raise Exception(f"x and y must be of same length or both more than N(={N}). FYI: len(x) = {len(x)}, len(y) = {len(y)}")
    if N > len(x):
        raise Exception(f"Don't have that many data points. Solution: reduce N to <= {len(x)}")
    if p < min(x[:N]) or p > max(x[:N]):
        raise Exception(f"Cannot interpolate for p = {p} outside of range [{min(x)}, {max(x)}]")
    s = []
    for i in range(N):
        pr = [(p-x[k])/(x[i]-x[k]) for k in range(N) if k != i]
        s.append(product(pr)*y[i])
    return sum(s)