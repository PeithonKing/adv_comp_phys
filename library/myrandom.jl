mutable struct MyRandom
    seed::Float64
    range::Vector{Float64}
end

function MyRandom(; seed=0.1, range=[0, 1])
    return MyRandom(seed, range)
end

function scale(rng::MyRandom, x)
    return rng.range[1] + x * (rng.range[2] - rng.range[1])
end

function rand(rng::MyRandom; a=1103515245, c=12345, m=32768)
    rng.seed = (a * rng.seed + c) % m
    return scale(rng, rng.seed / m)
end