module Integration

include("utils.jl")
include("equation.jl")

export simpson_rule
function simpson_rule(func, ll, ul, n=10)

    h = (ul - ll) / n
    s = 0

    for x in ll:h:ul-h
		# println(x)
        s += func(x) + 4 * func(x + h / 2) + func(x + h)
    end

    return s * h / 6
end

# export find_legendre_roots
function find_legendre_roots(n)
    roots = []
    for i in 1:n
        x0 = cos((2*i - 1) * π / (2 * n))  # Initial guess using Chebyshev nodes
        root = Eqn.solve_newton_raphson(x -> Utils.legendre(x, n), x -> Utils.legendre_derivative(x, n), x0)
        if !isnothing(root)
            push!(roots, root)
        end
    end
    return roots
end

export gaussian_quadrature
function gaussian_quadrature(f, a, b, n)
    # if n < 1 || !isa(n, Int)
    #     throw(ArgumentError("n must be a positive integer"))
    # end
    
    roots = find_legendre_roots(n)
    
    # Calculate the weights
    weights = [2 / ((1 - x^2) * Utils.legendre_derivative(x, n)^2) for x in roots]
    
    integral = 0
    for i in 1:n
        integral += weights[i] * f(((b - a) * roots[i])/2 + (b + a) / 2)
    end
    
    return (b - a) / 2 * integral
end




end





