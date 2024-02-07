module Utils

export differentiate
function differentiate(f, x, h=1e-6)
    """
    Compute the derivative of a function f at a point x using the central difference method.

    Args:
    - f (Function): Function to differentiate.
    - x (Float64): Point at which to differentiate the function.
    - h (Float64, optional): Step size. Defaults to 1e-6.

    Returns:
    - Float64: Derivative of the function at the point x.
    """
    return (f(x + h/2) - f(x - h/2)) / h
end

export truncate_to_decimal_places
function truncate_to_decimal_places(number, decimal_places)
    """
    Truncate a number to a specified number of decimal places.

    Args:
    - number (Float64): Number to truncate.
    - decimal_places (Int): Number of decimal places to truncate to.

    Returns:
    - Float64: Truncated number.
    """
    scale = 10 ^ decimal_places
    truncated_number = floor(number * scale) / scale
    return truncated_number
end

export legendre
function legendre(x, n)
    # Ensure n is a non-negative integer
    if n < 0 || n isa AbstractFloat
        throw(ArgumentError("n must be a non-negative integer"))
    end

    # Ensure x is in the range [-1, 1]
    if any(x .< -1) || any(x .> 1)
        throw(ArgumentError("x must be in the range [-1, 1]"))
    end

    if n == 0
        return 1.0
    elseif n == 1
        return x
    else
        return ((2n - 1) * x * legendre(x, n-1) - (n-1) * legendre(x, n-2)) / n
    end
end

function legendre_derivative(x, n, h = 1e-6)
    # if x == 1.0 || x == -1.0
    #     throw(ArgumentError("x cannot be 1 or -1"))
    # end
    return (legendre.(x+h/2, n) - legendre.(x-h/2, n)) / h
end

end