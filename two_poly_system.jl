#!/usr/bin/env julia
# two_poly_sampling_with_coeffs.jl
#
# Monte Carlo sampling of sparse 2-polynomial systems.
# Saves solution statistics AND full coefficient matrix C per trial.

using Random
using DynamicPolynomials
using HomotopyContinuation
using DataFrames
using CSV
using Dates

# ----------------------------
# Support + construction
# ----------------------------

function build_support(n::Int)
    n ≥ 2 || error("Require n ≥ 2.")
    A = Tuple{Int,Int}[]
    push!(A, (1, 0))
    push!(A, (0, 1))
    push!(A, (1, 1))
    for k in 2:n
        push!(A, (1, k))
    end
    push!(A, (0, 0))
    return A                  # length = n + 3
end

function sample_C(rng::AbstractRNG, m::Int)
    C = rand(rng, 2, m) .* 2 .- 1
    C[1,1] = 1.0
    C[1,2] = 0.0
    C[2,1] = 0.0
    C[2,2] = 1.0
    return C
end

function poly_from_support(c, A, x, y)
    p = zero(x)
    @inbounds for j in eachindex(A)
        a, b = A[j]
        p += c[j] * x^a * y^b
    end
    return p
end

function count_real_pos(sols; tol_real=1e-8, tol_pos=1e-10)
    nreal = 0
    npos  = 0
    for s in sols
        xr, xi = real(s[1]), imag(s[1])
        yr, yi = real(s[2]), imag(s[2])
        if abs(xi) ≤ tol_real && abs(yi) ≤ tol_real
            nreal += 1
            if xr ≥ tol_pos && yr ≥ tol_pos
                npos += 1
            end
        end
    end
    return nreal, npos
end

# ----------------------------
# Main experiment
# ----------------------------

function run_experiment(; n=5, N=10_000, seed=0)
    rng = MersenneTwister(seed)
    A = build_support(n)
    m = length(A)

    @polyvar x y

    # ---- DataFrame schema ----
    df = DataFrame(
        trial = Int[],
        success = Bool[],
        nsolutions = Int[],
        nreal = Int[],
        npos = Int[],
        time_sec = Float64[]
    )

    # coefficient columns
    for i in 1:2, j in 1:m
        df[!, Symbol("c$(i)_$(j)")] = Float64[]
    end

    println("Running N=$N trials with n=$n")
    println("Support size = $m")
    start_time = now()

    for t in 1:N
        C = sample_C(rng, m)
        p1 = poly_from_support(view(C,1,:), A, x, y)
        p2 = poly_from_support(view(C,2,:), A, x, y)
        F = System([p1, p2])

        t0 = time()
        try
            res = solve(F; start_system=:polyhedral)
            sols = solutions(res)
            nreal, npos = count_real_pos(sols)

            row = Dict(
                :trial => t,
                :success => true,
                :nsolutions => length(sols),
                :nreal => nreal,
                :npos => npos,
                :time_sec => time() - t0
            )

            for i in 1:2, j in 1:m
                row[Symbol("c$(i)_$(j)")] = C[i,j]
            end

            push!(df, row)

        catch err
            row = Dict(
                :trial => t,
                :success => false,
                :nsolutions => 0,
                :nreal => 0,
                :npos => 0,
                :time_sec => time() - t0
            )

            for i in 1:2, j in 1:m
                row[Symbol("c$(i)_$(j)")] = C[i,j]
            end

            push!(df, row)
        end

        if t % 100 == 0
            println("  progress: $t / $N")
        end
    end

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out = "two_poly_n$(n)_trials$(N)_with_coeffs_$timestamp.csv"
    CSV.write(out, df)

    println("\nFinished.")
    println("Saved to: $out")
    println("Total runtime: ", now() - start_time)

    return df
end

# ----------------------------
# CLI
# ----------------------------

function main()
    n = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 5
    N = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 10_000
    run_experiment(n=n, N=N)
end

main()