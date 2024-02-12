def mysum(a):
    """iteratively adds all elements of a list

    Args:
        a (iterable): list of numbers
    """
    try:
        total = a[0]
        for i in a[1:]:
            total += i
        return total
    except:
        return 0


def product(a):
    """iteratively multiplies all elements of a list

    Args:
        a (iterable): list of numbers
    """
    if len(a) == 0:
        raise Exception("Empty list, cannot calculate product")
    total = 1
    for i in a:
        total *= i
    return total

def differentiate(f, x, epsilon = 1e-6):  # Numerical differentiation
    return (f(x+epsilon)-f(x))/epsilon