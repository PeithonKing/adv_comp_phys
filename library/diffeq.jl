module Diffeq

include("utils.jl")
using Random



function rk4(f, x, x0, y0, h=0.01)
    """
    Fourth-order Runge-Kutta method for solving ODEs.

    Args:
        f (function): Function to be integrated.
        x (Float64): x value to be evaluated.
        x0 (Float64): Initial x value.
        y0 (Float64): Initial y value.
        h (Float64): Step size.

    Returns:
        Tuple{Vector{Float64}, Vector{Float64}, Float64}: Approximate solution to the ODE.
    """
    xs, ys = Float64[x0], Float64[y0]
    while x0 < x
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + h / 2, y0 + k1 / 2)
        k3 = h * f(x0 + h / 2, y0 + k2 / 2)
        k4 = h * f(x0 + h, y0 + k3)
        y0 += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x0 += h
        push!(xs, x0)
        push!(ys, y0)
    end
    return xs, ys, y0
end





end
