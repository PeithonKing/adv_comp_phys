module Eqn

include("utils.jl")
using Random



export fixed_point_iteration
function fixed_point_iteration(phi, x0, max_it=100, tolerance=1e-4)
    """
    Fixed-point iteration method for solving equations of the form x = phi(x).

    Parameters:
    - phi (Function): Function defining the fixed-point iteration.
    - x0 (Float64): Initial guess.
    - max_it (Int, optional): Maximum number of iterations. Defaults to 100.
    - tolerance (Float64, optional): Convergence tolerance. Defaults to 1e-4.

    Returns:
    - Float64: Approximate solution of the equation.

    Raises:
    - AssertionError: If the derivative of the function at the fixed point is greater than or equal to 1.
    """

    @assert abs(Utils.differentiate(phi, x0)) < 1 "The derivative of the function at the fixed point is greater than or equal to 1. The fixed-point iteration may not converge."

    for i in 1:max_it
        x = phi(x0)
        # println("Iteration $i: x = $x")
        if abs(x - x0) < tolerance
            return x
        end
        x0 = x
    end

    return x0
end

export solve_newton_raphson
function solve_newton_raphson(f, f_d = nothing, guess = nothing, delta=1e-4, rec_depth = 0, verbose=false)

    # println("lalalala la")

    if isnothing(guess)
        guess = rand(1)[1]  # multiply just for scaling
    end

    if isnothing(f_d)
        # @warn "No derivative provided, using numerical differentiation"
        f_d = x -> Utils.differentiate(f, x)
    end

    guess -= f(guess) / f_d(guess)

    if verbose
        println("step=$rec_depth\t  x=$guess\tf(x)=$(f(guess)))")
    end

    return (abs(f(guess)) > delta) ? solve_newton_raphson(f, f_d, guess, delta, rec_depth + 1, verbose) : guess
end



end
