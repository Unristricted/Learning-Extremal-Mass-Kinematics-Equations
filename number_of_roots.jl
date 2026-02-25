using HomotopyContinuation
using Random
using Distributions
using JSON3
using Dates
using Printf
using Statistics

function write_config_AC(path::AbstractString, A::Vector{Vector{Int}}, C::Matrix{Float64})
    config = (
        A=A,   # exponent vectors, length t, each a d-dimensional vector
        C=C    # s×t coefficient matrix
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

# ---- Build monomials in HC variables from exponent vectors ----
function build_monomials(vars::AbstractVector, exps::Vector{Vector{Int}})
    d = length(vars)
    @assert all(length(e) == d for e in exps)
    [prod(vars[i]^e[i] for i in 1:d) for e in exps]
end

function select_exponents(d::Int, n::Int, t_policy=:cap, t_max::Int=128, replacement::Bool=false, rng::AbstractRNG=Random.GLOBAL_RNG)
    A = monomial_exponents(d, n)
    N = length(A)
    t = begin
        if t_policy === :all
            N
        elseif t_policy === :cap
            min(N, t_max)
        elseif t_policy isa Integer
            min(N, t_policy)
        else
            error("Unsupported t_policy = $t_policy")
        end
    end
    if t == N
        return A, (t, N, false)
    else
        @assert replacement || t <= N "t must be <= N without replacement"
        idx = replacement ? rand(rng, 1:N, t) : first(randperm(rng, N), t)
        return A[idx], (t, N, true)
    end
end

function build_system(d::Int, n::Int, s::Int, t_policy=:cap, t_max::Int=128, replacement::Bool=false, seed::Int)
    Random.seed!(seed)
    @var x[1:d]

    exps_sel, (t, N, sampled) = select_exponents(d, n; t_policy=t_policy, t_max=t_max, replacement=replacement, rng=Random.default_rng())

    mons = build_monomials(x, exps_sel)

    C = rand(Uniform(-1, 1), s, length(exps_sel))
    polys = [sum(C[j, k] * mons[k] for k in 1:length(exps_sel)) for j in 1:s]
    F = System(polys; variables=x)
    return F, x, exps_sel, C, length(exps_sel), N, sampled
end

function run_trial(d::Int, n::Int, s::Int, trial::Int; t_policy=:cap, t_max::Int=128, replacement::Bool=false, seed_base::Int=12345, solver_kwargs=(start_system=:polyhedral,),
    save_A::Bool=true, save_C::Bool=true)

    seed = Int(mod(hash((d, n, s, trial, seed_base)), typemax(Int)))

    t_build = 0.0
    F_ = nothing
    x_ = nothing
    A_ = nothing
    C_ = nothing
    t_ = 0
    N_ = 0
    sampled_ = false
    t_build = @elapsed begin
        F, x, A, C, t, N, sampled = build_system(d, n, s;
            t_policy=t_policy, t_max=t_max,
            replacement=replacement, seed=seed)
        F_, x_, A_, C_, t_, N_, sampled_ = F, x, A, C, t, N, sampled
    end

    ok = true
    err = ""
    num_real = -1
    t_solve = 0.0
    try
        t_solve = @elapsed begin
            res = solve(F_; solver_kwargs...)
            num_real = length(real_solutions(res; only_finite=true))
        end
    catch e
        ok = false
        err = sprint(showerror, e)
    end

    t_total = t_build + t_solve

    payload = Dict(
        "trial" => trial,
        "timestamp" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ"),
        "ok" => ok,
        "error" => err,
        "metrics" => Dict(
            "num_real_roots" => num_real,
            "t_build_sec" => t_build,
            "t_solve_sec" => s_solve,
            "t_total_sec" => t_total
        ),
        "A" => save_A ? A_ : nothing,
        "C" => save_C ? C_ : nothing,
        "t" => t_,
        "N_total_monomials" => N_,
        "sampled_monomials" => sampled_,
        "seed" => seed
    )
    return payload, ok
end

function run_group_and_save!(outdir::AbstractString, d::Int, n::Int, s::Int, TRIALS::Int;
    t_policy=:cap, t_max::Int=128, replacement::Bool=false,
    seed_base::Int=12345, solver_kwargs=(start_system=:polyhedral,),
    save_A::Bool=true, save_C::Bool=true)

    is_dir(outdir) || mkpath(outdir)

    N = count_monomials(d, n)
    t = t_policy === :all ? N :
        t_policy === :cap ? min(N, t_max) :
        t_policy isa Integer ? min(N, t_policy) :
        error("Unsupported t_policy = $t_policy")

    trials = Vector{Dict}(undef, TRIALS)
    failures = 0
    t0 = time()

    for trial in 1:TRIALS
        payload, ok = run_trial(d, n, s, trial;
            t_policy=t_policy, t_max=t_max, replacement=replacement,
            seed_base=seed_base, solver_kwargs=solver_kwargs,
            save_A=save_A, save_C=save_C)

        trial[trial] = payload
        failures += ok ? 0 : 1
    end

    elapsed = time() - t0

    ok_mask = [get(tr, "ok", false) for tr in trials]
    ok_trials = [tr for (tr, ok) in zip(trials, ok_mask) if ok]
    num_ok = length(ok_trials)
    num_real_vals = [tr["metrics"]["num_real_roots"] for tr in ok_trials]
    t_solve_vals = [tr["metrics"]["t_solve_sec"] for tr in ok_trials]
    t_total_vals = [tr["metrics"]["t_total_sec"] for tr in ok_trials]

    summary = Dict(
        "runs" => TRIALS,
        "failures" => failures,
        "successes" => num_ok,
        "wall_elapsed_sec" => elapsed,
        "num_real_roots" => Dict(
            "mean" => num_ok > 0 ? mean(num_real_vals) : nothing,
            "median" => num_ok > 0 ? median(num_real_vals) : nothing,
            "min" => num_ok > 0 ? minimum(num_real_vals) : nothing,
            "max" => num_ok > 0 ? maximum(num_real_vals) : nothing,
        ),
        "t_solve_sec" => Dict(
            "mean" => num_ok > 0 ? mean(t_solve_vals) : nothing,
            "median" => num_ok > 0 ? median(t_solve_vals) : nothing
        ),
        "t_total_sec" => Dict(
            "mean" => num_ok > 0 ? mean(t_total_vals) : nothing,
            "median" => num_ok > 0 ? median(t_total_vals) : nothing
        )
    )

    group_payload = Dict(
        "timestamp" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS.sssz"),
        "group_params" => Dict(
            "d" => d, "n" => n, "s" => s, "TRIALS" => TRIALS,
            "t" => t, "N_total_monomials" => N,
            "replacement" => replacement,
            "t_policy" => t_policy === :cap ? "cap" : t_policy === :all ? "all" : string(t_policy),
            "t_max" => t_max,
            "seed_base" => seed_base,
            "solver_kwargs" => Dict(string(k) => string(v) for (k, v) in pairs(NamedTuple(solver_kwargs))),
            "save_A" => save_A,
            "save_C" => save_C
        ),
        "summary" => summary,
        "trials" => trials
    )

    filename = @sprintf("d%d_n%d_s%d_t%d_trials%d_seedbase%d_rep%d.json", d, n, s, t, TRIALS, seed_base, replacement ? 1 : 0)
    path = joinpath(outdir, filename)
    open(path, "w") do io
        JSON3.write(io, group_payload; indent=2)
    end
    return path, failures == 0
end

function run_sweep(; DMAX::Int=20, NMAX::Int=20, SMAX::Int=20, TRIALS::Int=1000,
    outdir::AbstractString="results",
    t_policy=:cap, t_max::Int=128, replacement::Bool=false,
    seed_base::Int=12345,
    solver_kwargs=(start_system=:polyhedral,),
    save_A::Bool=true, save_C::Bool=true
)

    println("Starting sweep: d∈[1, $DMAX], n∈[0,$NMAX], s∈[1,$SMAX], trials/group=$TRIALS")
    println("t_policy=$(t_policy), t_max=$t_max, replacement=$(replacement)")
    println("Writing group files to: $outdir")
    total_groups = (DMAX) * (NMAX + 1) * (SMAX)
    println("Planned Groups: $total_groups")

    groups = 0
    group_failures = 0
    t0 = time()

    for d in 1:DMAX, n in 0:NMAX, s in 1:SMAX
        groups += 1
        path, ok_group = run_group_and_save!(outdir, d, n, s, TRIALS;
            t_policy=t_policy, t_max=t_max, replacement=replacement,
            seed_base=seed_base, solver_kwargs=solver_kwargs,
            save_A=save_A, save_C=save_C)

        ok_group || (group_failures += 1)

        if groups % 10 == 0
            elapsed = time() - t0
            @printf("[groups %6d / %6d] failed_groups=%d elapsed=%.1fs last=%s\n",
                groups, total_groups, group_failures, elapsed, path)
        end
    end

    elapsed = time() - t0
    println("Done. Groups: $groups, Failed groups: $group_failures, Elapsed: $(round(elapsed, digits=1)) s")
end

if abspath(PROGRAM_FILE) == @__FILE__
    DMAX = 10
    NMAX = 10
    SMAX = 10
    TRIALS = 100

    OUTDIR = "results1"

    T_POLICY = :cap
    T_MAX = 128
    REPLACEMENT = false

    SEED_BASE = 20251106

    SOLVER_KW = (start_system=:polyhedral,)

    SAVE_A = true
    SAVE_C = true

    run_sweep(; DMAX=DMAX, NMAX=NMAX, SMAX=SMAX, TRIALS=TRIALS,
        outdir=OUTDIR, t_policy=T_POLICY, t_max=T_MAX, replacement=REPLACEMENT,
        seed_base=SEED_BASE, solver_kwargs=SOLVER_KW, save_A=SAVE_A, save_C=SAVE_C)
end
