# wnt_sampling.jl
#
# Wnt shuttle model (arXiv:1502.03188) steady-state sampling experiment.
# - Encode 14 non-redundant steady-state polynomials + 5 conservation laws = 19 equations in 19 unknowns.
# - Sample parameters randomly (choose sampling box below)
# - Solve with HomotopyContinuation.jl
# - Count: total solutions returned, real solutions, positive real solutions
# - Repeat N times and save CSV
#
# NOTE:
# - "Positive" means: solution is real within tol_real AND all coordinates >= tol_pos
# - You can change positivity to require x_i > 0 only for a subset if desired.

using HomotopyContinuation
using Random
using DataFrames
using CSV
using Dates
using DynamicPolynomials   # needed for subs

# -----------------------
# Variables / parameters
# -----------------------
@var x[1:19]
@var k[1:31]
@var c[1:5]

# -----------------------
# Steady-state equations (14 generators)
# -----------------------
f1  = -k[1]*x[1] + k[2]*x[2]

f2  =  k[1]*x[1] - (k[2] + k[26])*x[2] + k[27]*x[3] - k[3]*x[2]*x[4] + (k[4] + k[5])*x[14]

f3  =  k[26]*x[2] - k[27]*x[3] - k[14]*x[3]*x[6] + (k[15] + k[16])*x[15]

f4  = -k[3]*x[2]*x[4] - k[9]*x[4]*x[10] + k[4]*x[14] + k[8]*x[16] + (k[10] + k[11])*x[18]

f5  = -k[28]*x[5] + k[29]*x[7] - k[6]*x[5]*x[8] + k[5]*x[14] + k[7]*x[16]

f6  = -k[14]*x[3]*x[6] - k[20]*x[6]*x[11] + k[15]*x[15] + k[19]*x[17] + (k[21] + k[22])*x[19]

f7  =  k[28]*x[5] - k[29]*x[7] - k[17]*x[7]*x[9] + k[16]*x[15] + k[18]*x[17]

f8  = -k[6]*x[5]*x[8] + (k[7] + k[8])*x[16]

f9  = -k[17]*x[7]*x[9] + (k[18] + k[19])*x[17]

f10 =  k[12] - (k[13] + k[30])*x[10] - k[9]*x[4]*x[10] + k[31]*x[11] + k[10]*x[18]

f11 = -k[23]*x[11] + k[30]*x[10] - k[31]*x[11] - k[20]*x[6]*x[11] - k[24]*x[11]*x[12] + k[25]*x[13] + k[21]*x[19]

f12 = -k[24]*x[11]*x[12] + k[25]*x[13]

f14 =  k[3]*x[2]*x[4] - (k[4] + k[5])*x[14]

f18 =  k[9]*x[4]*x[10] - (k[10] + k[11])*x[18]

# -----------------------
# Conservation laws (5)
# -----------------------
g1 = (x[1] + x[2] + x[3] + x[14] + x[15]) - c[1]
g2 = (x[4] + x[5] + x[6] + x[7] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19]) - c[2]
g3 = (x[8] + x[16]) - c[3]
g4 = (x[9] + x[17]) - c[4]
g5 = (x[12] + x[13]) - c[5]

polys  = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f14,f18,g1,g2,g3,g4,g5]
params = vcat(k, c)

# -----------------------
# Sampling configuration
# -----------------------
"""
Sample a vector uniformly from [lo, hi]^d.
"""
sample_box(rng::AbstractRNG, d::Int; lo::Float64, hi::Float64) = lo .+ (hi - lo) .* rand(rng, d)

# Choose sampling box here:
#   1) Symmetric: [-1, 1]
#   2) Positive (more "physical"): [0, 100]  (similar spirit to the paper's positive sampling)
const SAMPLE_MODE = :pos10   # :symm1 or :pos100

function sample_parameters(rng::AbstractRNG)
    if SAMPLE_MODE == :symm1
        θ = sample_box(rng, 36; lo=-1.0, hi=1.0)
    elseif SAMPLE_MODE == :pos100
        θ = sample_box(rng, 36; lo=0.0, hi=100.0)
    elseif SAMPLE_MODE == :pos1
        θ = sample_box(rng, 36; lo=0.0, hi=1.0)
    elseif SAMPLE_MODE == :pos10
        θ = sample_box(rng, 36; lo=0.0, hi=10.0)
    else
        error("Unknown SAMPLE_MODE = $SAMPLE_MODE. Use :symm1 or :pos100.")
    end
    kvals = θ[1:31]
    cvals = θ[32:36]
    return kvals, cvals, vcat(kvals, cvals)
end

# -----------------------
# System instantiation at parameter point
# -----------------------
function concrete_system(polys, params, parvals, x)
    pdict  = Dict(params[i] => parvals[i] for i in eachindex(params))
    polysθ = subs.(polys, Ref(pdict))
    return System(polysθ; variables=x)
end

# -----------------------
# Counting helpers
# -----------------------
"""
Count real solutions among solutions(res) using tol_real on imaginary parts.
Works whether solutions(res) returns vectors or Solution objects.
"""
function count_real_solutions(res; tol_real=1e-7)
    nreal = 0
    for s in solutions(res)
        z = (hasproperty(s, :u) ? s.u : s)  # support both representations
        if maximum(abs.(imag.(z))) <= tol_real
            nreal += 1
        end
    end
    return nreal
end

"""
Count positive real solutions:
- real within tol_real
- all coordinates >= tol_pos (on real parts)
"""
function count_positive_solutions(res; tol_real=1e-7, tol_pos=1e-9)
    npos = 0
    for s in solutions(res)
        z = (hasproperty(s, :u) ? s.u : s)
        if maximum(abs.(imag.(z))) <= tol_real && minimum(real.(z)) >= tol_pos
            npos += 1
        end
    end
    return npos
end

# -----------------------
# Experiment runner
# -----------------------
function run_experiment(; N=10_000,
                        seed=1,
                        tol_real=1e-7,
                        tol_pos=1e-9,
                        start_system=:polyhedral,   # or :total_degree
                        out_csv="wnt_samples.csv")

    rng = MersenneTwister(seed)

    # DataFrame columns
    cols = Dict{Symbol, Any}()
    cols[:trial] = Int[]
    cols[:ok] = Bool[]
    cols[:n_solutions] = Int[]
    cols[:n_real] = Int[]
    cols[:n_real_library] = Int[]   # for comparison with count_real_solutions(res)
    cols[:n_pos] = Int[]
    for i in 1:31; cols[Symbol("k$i")] = Float64[]; end
    for i in 1:5;  cols[Symbol("c$i")] = Float64[]; end
    cols[:err] = String[]
    df = DataFrame(cols)

    t0 = now()
    println("Starting N=$N trials at $(t0). Output: $out_csv")
    println("SAMPLE_MODE=$(SAMPLE_MODE) | start_system=$(start_system) | tol_real=$(tol_real) | tol_pos=$(tol_pos)")

    for t in 1:N
        kvals, cvals, parvals = sample_parameters(rng)

        ok    = true
        nsol  = 0
        nreal = 0
        nreal_library = 0
        npos  = 0
        err   = ""

        try
            Fθ  = concrete_system(polys, params, parvals, x)
            res = solve(Fθ; start_system=start_system)

            if t == 1
                println("solutions(res) element type: ", typeof(first(solutions(res))))
            end
            nsol  = length(solutions(res))
            nreal = count_real_solutions(res; tol_real=tol_real)
            nreal_library = length(real_solutions(res; only_finite=true))
            npos  = count_positive_solutions(res; tol_real=tol_real, tol_pos=tol_pos)

        catch e
            ok = false
            bt = catch_backtrace()
            err = sprint(showerror, e, bt)
            if t <= 3
                println("\nFirst failure at trial $t:\n$err\n")
            end
        end

        # Save row
        push!(df.trial, t)
        push!(df.ok, ok)
        push!(df.n_solutions, nsol)
        push!(df.n_real, nreal)
        push!(df.n_real_library, nreal_library)
        push!(df.n_pos, npos)
        for i in 1:31; push!(df[!, Symbol("k$i")], kvals[i]); end
        for i in 1:5;  push!(df[!, Symbol("c$i")], cvals[i]); end
        push!(df.err, err)

        if t % 100 == 0
            println("trial $t / $N | ok=$(count(df.ok)) | last: nsol=$nsol, nreal=$nreal, npos=$npos")
        end
    end

    CSV.write(out_csv, df)
    println("Done. Wrote $(nrow(df)) rows to $out_csv")
    println("Finished at $(now()). Elapsed: $(now() - t0)")
    return df
end

# Run as script
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment(N=10_000,
                   seed=1,
                   tol_real=1e-7,
                   tol_pos=1e-9,
                   start_system=:polyhedral,
                   out_csv="wnt_samples_10k_10.csv")
end