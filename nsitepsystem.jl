# phospho_sampling_1904_11633.jl
#
# arXiv:1904.11633 (G_J subnetwork) steady-state sampling experiment.
# - Distributive sequential n-site phosphorylation/dephosphorylation with intermediates only on E-side:
#     If j ∈ J:   S_j + E <-> Y_j -> S_{j+1} + E
#     If j ∉ J:   S_j + E -> S_{j+1} + E   (no intermediate)
#     For all j:  S_{j+1} + F -> S_j + F   (no intermediate)
# - Build mass-action ODEs, set steady-state equations, add conservation laws:
#     sum(S_i) + sum(Y_j) = Stot
#     E + sum(Y_j) = Etot
#     F = Ftot
# - Sample parameters uniformly from [-1, 1]
# - Solve with HomotopyContinuation.jl, count number of real roots
# - Repeat N times and save CSV
#
# Run:
#   julia --project -e 'import Pkg; Pkg.add(["HomotopyContinuation","DynamicPolynomials","DataFrames","CSV","Dates"])'
#   julia phospho_sampling_1904_11633.jl
#
# You can edit n and Jset below.

using HomotopyContinuation
using Random
using DataFrames
using CSV
using Dates
using DynamicPolynomials  # for subs

# -----------------------
# User configuration
# -----------------------
const N_DEFAULT = 10_000
const SEED_DEFAULT = 1

# Choose the phosphorylation length "n" (sites); substrates are S0..Sn
const N_SITES = 3

# Choose J ⊆ {0,1,...,n-1} for which E-side intermediates Y_j exist.
# Example: full E intermediates => J = 0:(n-1)
const Jset = collect(0:(N_SITES-1))

# Sampling box:
const SAMPLE_LO = -1.0
const SAMPLE_HI =  1.0

# Tolerances for "real"
const TOL_REAL = 1e-7

# HomotopyContinuation start system
const START_SYSTEM = :polyhedral  # or :total_degree

# Output file
const OUT_CSV = "phospho_1904_11633_samples_10k.csv"

# -----------------------
# Helpers
# -----------------------
sample_box(rng::AbstractRNG, d::Int; lo::Float64, hi::Float64) =
    lo .+ (hi - lo) .* rand(rng, d)

"""
Count real solutions using imag-part tolerance.
"""
function count_real_solutions(res; tol_real=1e-7)
    nreal = 0
    for s in solutions(res)
        z = (hasproperty(s, :u) ? s.u : s)  # support Solution object or raw vector
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
# Build polynomial steady-state system for G_J
# -----------------------
"""
Build steady-state polynomial system for arXiv:1904.11633 network G_J.

Variables:
  s[1:n+1]  corresponds to S0..Sn
  e         kinase E
  f         phosphatase F
  y[1:|J|]  corresponds to Y_j, j ∈ J in increasing order

Parameters (all sampled):
  For each j ∈ J: kon_j, koff_j, kcat_j
  For each j ∉ J: tau_j
  For each j=0..n-1: nu_j
  Totals: Stot, Etot, Ftot

Equations:
  ds_i/dt = 0 for i = 0..n-1   (i.e., s[1]..s[n])
  dy_j/dt = 0 for each j ∈ J
  conservation: sum(S)+sum(Y)-Stot=0,  e+sum(Y)-Etot=0,  f-Ftot=0
Returns:
  polys :: Vector{Polynomial}
  vars  :: Vector{Variable}
  params:: Vector{Variable}
  meta  :: NamedTuple with parameter names/order
"""
# -----------------------
# Build polynomial steady-state system for G_J  (NO Variable type usage)
# -----------------------
function build_system_GJ(n::Int, J::Vector{Int})
    J_sorted = sort(unique(J))
    Jset_local = Set(J_sorted)

    # Variables
    @var s[1:(n+1)]  # S0..Sn
    @var e f
    @var y[1:length(J_sorted)]  # Y_j for j ∈ J (ordered)

    vars = vcat(s, [e, f], y)

    # Parameter bookkeeping (no explicit Variable types)
    pnames = String[]
    params = Any[]

    kon  = Dict{Int,Any}()
    koff = Dict{Int,Any}()
    kcat = Dict{Int,Any}()
    tau  = Dict{Int,Any}()
    nu   = Dict{Int,Any}()

    # Count parameters:
    nJ = length(J_sorted)
    nNotJ = n - nJ
    P = 3*nJ + nNotJ + n + 3  # (kon,koff,kcat for J) + tau for notJ + nu for all + totals 3
    @var p[1:P]
    k = 1

    # J parameters
    for j in J_sorted
        kon[j]  = p[k];  push!(params, p[k]); push!(pnames, "kon_$j");  k += 1
        koff[j] = p[k];  push!(params, p[k]); push!(pnames, "koff_$j"); k += 1
        kcat[j] = p[k];  push!(params, p[k]); push!(pnames, "kcat_$j"); k += 1
    end

    # not-J phosphorylation direct rates
    for j in 0:(n-1)
        if !(j in Jset_local)
            tau[j] = p[k]; push!(params, p[k]); push!(pnames, "tau_$j"); k += 1
        end
    end

    # dephosphorylation rates nu_j for all j
    for j in 0:(n-1)
        nu[j] = p[k]; push!(params, p[k]); push!(pnames, "nu_$j"); k += 1
    end

    # totals
    Stot = p[k]; push!(params, p[k]); push!(pnames, "Stot"); k += 1
    Etot = p[k]; push!(params, p[k]); push!(pnames, "Etot"); k += 1
    Ftot = p[k]; push!(params, p[k]); push!(pnames, "Ftot"); k += 1

    @assert k == P + 1

    # Map j -> y variable
    y_of = Dict{Int,Any}()
    for (idx, j) in enumerate(J_sorted)
        y_of[j] = y[idx]
    end

    # Initialize derivatives (avoid zero(...) ambiguity)
    ds = [0*s[1] for _ in 1:(n+1)]
    de = 0*e
    df = 0*f
    dy = Dict{Int,Any}()
    for j in J_sorted
        dy[j] = 0*y_of[j]
    end

    # Apply reaction updates
    function add_reaction!(rate, Δs::Dict{Int,Int}, Δe::Int, Δf::Int, Δy::Dict{Int,Int})
        # substrates S_i are stored as s[i+1] where i=0..n
        for (i, coeff) in Δs
            ds[i+1] += coeff * rate
        end
        de += Δe * rate
        df += Δf * rate
        for (j, coeff) in Δy
            dy[j] += coeff * rate
        end
        return nothing
    end

    # Build reactions for each j = 0..n-1
    for j in 0:(n-1)
        # Dephosph: S_{j+1} + F -> S_j + F, rate nu_j * S_{j+1} * F
        rate_deph = nu[j] * s[(j+1)+1] * f
        add_reaction!(rate_deph,
                      Dict(j => +1, (j+1) => -1),
                      0, 0,
                      Dict{Int,Int}())

        # Phosphorylation step j
        if j in Jset_local
            # S_j + E -> Y_j
            rate_bind = kon[j] * s[j+1] * e
            add_reaction!(rate_bind,
                          Dict(j => -1),
                          -1, 0,
                          Dict(j => +1))

            # Y_j -> S_j + E
            rate_unbind = koff[j] * y_of[j]
            add_reaction!(rate_unbind,
                          Dict(j => +1),
                          +1, 0,
                          Dict(j => -1))

            # Y_j -> S_{j+1} + E
            rate_cat = kcat[j] * y_of[j]
            add_reaction!(rate_cat,
                          Dict((j+1) => +1),
                          +1, 0,
                          Dict(j => -1))
        else
            # direct: S_j + E -> S_{j+1} + E (E cancels)
            rate_direct = tau[j] * s[j+1] * e
            add_reaction!(rate_direct,
                          Dict(j => -1, (j+1) => +1),
                          0, 0,
                          Dict{Int,Int}())
        end
    end

    # Steady-state equations:
    polys = Any[]

    # Use ds for S0..S_{n-1} (omit last substrate equation to square with conservation laws)
    for i in 1:n
        push!(polys, ds[i])
    end
    # dy_j = 0 for each j ∈ J
    for j in J_sorted
        push!(polys, dy[j])
    end
    # Conservation laws (paper (1.4)):
    push!(polys, sum(s) + sum(y) - Stot)   # total substrate
    push!(polys, e + sum(y) - Etot)        # total kinase
    push!(polys, f - Ftot)                 # total phosphatase

    meta = (pnames=pnames, J=J_sorted, n=n, P=P)
    return polys, vars, params, meta
end

# -----------------------
# Substitute numeric parameter values into symbolic polynomials and return HC System
# -----------------------
function concrete_system(polys, params, parvals, vars)
    pdict  = Dict(params[i] => parvals[i] for i in eachindex(params))
    polysθ = subs.(polys, Ref(pdict))
    return System(polysθ; variables=vars)
end

# -----------------------
# Experiment runner
# -----------------------
function run_experiment(; N::Int=N_DEFAULT,
                        seed::Int=SEED_DEFAULT,
                        tol_real::Float64=TOL_REAL,
                        n::Int=N_SITES,
                        J::Vector{Int}=Jset,
                        out_csv::String=OUT_CSV,
                        start_system=START_SYSTEM)

    rng = MersenneTwister(seed)

    polys, vars, params, meta = build_system_GJ(n, J)
    P = meta.P

    # DataFrame columns
    cols = Dict{Symbol, Any}()
    cols[:trial] = Int[]
    cols[:ok] = Bool[]
    cols[:n_solutions] = Int[]
    cols[:n_real] = Int[]
    cols[:n_pos] = Int[]
    for nm in meta.pnames
        cols[Symbol(nm)] = Float64[]
    end
    cols[:err] = String[]
    df = DataFrame(cols)

    t0 = now()
    println("Starting N=$N trials at $(t0). Output: $out_csv")
    println("n=$n | J=$(meta.J) | P=$P params sampled in [$SAMPLE_LO, $SAMPLE_HI] | start_system=$start_system | tol_real=$tol_real")

    for t in 1:N
        parvals = sample_box(rng, P; lo=SAMPLE_LO, hi=SAMPLE_HI)

        ok = true
        nsol = 0
        nreal = 0
        npos = 0
        err = ""

        try
            Fθ = concrete_system(polys, params, parvals, vars)
            res = solve(Fθ; start_system=start_system)

            if t == 1 && !isempty(solutions(res))
                println("solutions(res) element type: ", typeof(first(solutions(res))))
            end

            nsol = length(solutions(res))
            nreal = count_real_solutions(res; tol_real=tol_real)
            npos = count_positive_solutions(res; tol_real=tol_real, tol_pos=tol_real)

        catch e
            ok = false
            bt = catch_backtrace()
            err = sprint(showerror, e, bt)
            if t <= 3
                println("\nFirst failure at trial $t:\n$err\n")
            end
        end

        # Save row
        for (i, nm) in enumerate(meta.pnames)
            push!(df[!, Symbol(nm)], parvals[i])
        end
        push!(df.trial, t)
        push!(df.ok, ok)
        push!(df.n_solutions, nsol)
        push!(df.n_real, nreal)
        push!(df.n_pos, npos)
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
    run_experiment()
end