# include utils.js
include("utils.jl")

function fixed_point_iteration(func, x0, max_it=100, tolerance=1e-4)
    @assert abs(differentiate(func, x0)) < 1 "The derivative of the function at the fixed point is greater than or equal to 1. The fixed-point iteration may not converge."
    
    for i in 1:max_it
        x = func(x0)
        # println("Iteration $i: x = $x")
        if abs(x - x0) < tolerance
            return x
        end
        x0 = x
    end
    
    return x0
end
