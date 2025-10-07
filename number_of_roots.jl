using HomotopyContinuation
using Random
using JSON3

function write_config_AC(path::AbstractString, A::Vector{Vector{Int}}, C::Matrix{Float64})
    config = (
        A = A,   # exponent vectors, length t, each a d-dimensional vector
        C = C    # s×t coefficient matrix
    )
    open(path, "w") do io
        JSON3.write(io, config; indent=2)
    end
end

# ---- Exponent vectors (degree ≤ n) ----
function monomial_exponents(d::Integer, n::Integer)
    @assert d ≥ 1
    @assert n ≥ 0
    exps = Vector{Vector{Int}}()
    cur = zeros(Int, d)

    function assign_pos(pos::Int, remaining::Int)
        if pos == d
            cur[pos] = remaining
            push!(exps, copy(cur))
        else
            @inbounds for k in 0:remaining
                cur[pos] = k
                assign_pos(pos + 1, remaining - k)
            end
        end
    end

    for total in 0:n
        assign_pos(1, total)
    end
    return exps
end

count_monomials(d::Integer, n::Integer) = binomial(d + n, n)

# ---- Sample t exponent vectors uniformly (with/without replacement) ----
function sample_monomial_exps(d::Integer, n::Integer, t::Integer; replacement::Bool=false, seed=nothing)
    seed === nothing || Random.seed!(seed)
    exps = monomial_exponents(d, n)
    N = length(exps)
    @assert replacement || t ≤ N "t must be ≤ $(N) without replacement"
    idx = replacement ? rand(1:N, t) : first(randperm(N), t)
    return exps[idx], idx
end

# ---- Build monomials in HC variables from exponent vectors ----
function build_monomials(vars::AbstractVector, exps::Vector{Vector{Int}})
    d = length(vars)
    @assert all(length(e) == d for e in exps)
    [prod(vars[i]^e[i] for i in 1:d) for e in exps]
end

# ---- Random coefficient matrix in [0,1) ----
coefficient_matrix(s::Integer, t::Integer; seed=nothing) = (seed === nothing || Random.seed!(seed); rand(s, t))

# ---- Put it all together: random system of size s from t sampled monomials ----
"""
    random_system(d, n, s, t; replacement=false, seed=nothing)

Return:
  F :: System                # HomotopyContinuation system of length s
  x :: Vector{Variable}      # variables x[1:d]
  C :: Matrix{Float64}       # s×t coefficient matrix
  exps_sel :: Vector{Vector{Int}}  # sampled exponent vectors (length t)
  mons :: Vector{<:Any}      # the t monomials used (built from x)

Each equation is sum_{k=1..t} C[j,k] * mons[k]. If the constant monomial (all-zero exponents)
was sampled, that row includes a constant term.
"""
function random_system(d::Integer, n::Integer, s::Integer, t::Integer; replacement::Bool=false, seed=nothing)
    # Use a single seed for reproducibility across both sampling and coefficients
    seed === nothing || Random.seed!(seed)

    # variables
    @var x[1:d]

    # sample monomials
    exps_sel, _ = sample_monomial_exps(d, n, t; replacement=replacement)
    mons = build_monomials(x, exps_sel)

    println.(mons)
    # coefficients and equations
    C = rand(s, t)
    polys = [sum(C[j, k] * mons[k] for k in 1:t) for j in 1:s]

    F = System(polys; variables = x)
    return F, x, C, exps_sel, mons
end

function eval_at_point(F, x, p::AbstractVector)
    @assert length(x) == length(p) "Point dimension must match variables"
    pt = Dict(x[i] => p[i] for i in eachindex(x))
    return evaluate(F, pt)           # Vector of values F(p)
end

"""
    residual_report(F, x, res; tol=1e-10, print_values=true)

Evaluate F at each finite real solution from `res` (a `solve` result), print the
solution, F(p), and the infinity-norm residual. Flags any residuals > tol.
"""
function residual_report(F, x, res; tol::Real=1e-10, print_values::Bool=true)
    rsols = real_solutions(res; only_finite=false)
    println("Found $(length(rsols)) finite real solution(s).")

    for (j, p) in enumerate(rsols)
        vals = eval_at_point(F, x, p)
        maxabs = maximum(abs.(vals))
        println("\n— Solution $j —")
        println("x = ", p)
        if print_values
            println("F(x) = ", vals)
        end
        println("‖F(x)‖∞ = ", maxabs)
        if maxabs > tol
            println("WARN: residual above tol = $tol")
        end
    end
    return nothing
end

# ---- Demo ----
if abspath(PROGRAM_FILE) == @__FILE__
    # Settings
    d, n = 3, 2          # variables, max total degree
    s, t = 4, 6          # equations, sampled terms

    seed = rand(1:100000)

    F, x, C, exps_sel, mons = random_system(d, n, s, t; replacement=false, seed=seed)
    println(F)
    println("\nCoefficient matrix C ($(size(C))):\n", C)
    println("\nSampled exponent vectors:")
    #foreach(e -> println(e), exps_sel)
    println.(exps_sel)
    println("Equations:")
    println(F)

    # (Optional) try solving if square-ish / well-posed for your use case
    res = solve(F, start_system=:polyhedral)
    println(real_solutions(res, only_real=false, only_finite=false))
    println(length(real_solutions(res, only_real=false, only_finite=true)))

    residual_report(F, x, res; tol=1e-9)

    write_config_AC("poly_config.json", exps_sel, C)
    println("Config written to poly_config.json")
end
