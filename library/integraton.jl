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

export find_legendre_roots
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




end





