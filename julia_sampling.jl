using HomotopyContinuation
using Random
using CSV, DataFrames
using Dates

# --- you already built this earlier ---
# sys = System(F)

const D = 36  # 31 k's + 5 c's

"sample log-params uniformly in [lb,ub]"
sample_action(rng, lb, ub) = lb .+ (ub-lb) .* rand(rng, D)

"convert log-params to positive params"
function action_to_params(a::Vector{Float64})
    k = exp.(a[1:31])
    c = exp.(a[32:36])
    return k, c
end

"count real and positive-real solutions from an HC result"
function count_real_solutions(result; tol_im=1e-8, tol_pos=-1e-10)
    sols = solutions(result)
    n_real = 0
    n_pos  = 0
    for s in sols
        # HomotopyContinuation returns vectors of Complex
        if all(abs.(imag.(s)) .< tol_im)
            n_real += 1
            xr = real.(s)
            if all(xr .>= tol_pos)
                n_pos += 1
            end
        end
    end
    return n_real, n_pos
end

"evaluate one action; returns metrics"
function evaluate_action(sys, a; tol_im=1e-8, tol_pos=-1e-10)
    t0 = time()
    k, c = action_to_params(a)

    # parameter dictionary: IMPORTANT
    # This depends on how you named params (k[1]..k[31], c[1]..c[5]) in your System.
    # If you used @var k[1:31] @var c[1:5], then you can pass:
    p = Dict()
    for i in 1:31
        p[k[i]] = k[i]  # <-- this line is wrong if you shadow names; see note below.
    end
    for j in 1:5
        p[c[j]] = c[j]
    end

    # NOTE: Don't shadow k/c symbolic vars with numeric vectors.
    # Safer approach: keep symbolic arrays ks, cs:
    # @var ks[1:31] cs[1:5]
    # then in dict do p[ks[i]] = k_num[i], etc.

    try
        result = solve(sys; parameters=p)
        n_real, n_pos = count_real_solutions(result; tol_im=tol_im, tol_pos=tol_pos)
        runtime = time() - t0
        return (status="ok", n_real=n_real, n_pos=n_pos, runtime=runtime)
    catch err
        runtime = time() - t0
        return (status="fail", n_real=0, n_pos=0, runtime=runtime)
    end
end

function generate_dataset(sys; N=5000, lb=-6.0, ub=6.0, seed=1, outpath="roots_dataset.csv")
    rng = MersenneTwister(seed)
    rows = Vector{Any}(undef, N)

    for i in 1:N
        a = sample_action(rng, lb, ub)
        metrics = evaluate_action(sys, a)
        reward = metrics.n_real  # or metrics.n_pos
        rows[i] = (a=a, reward=reward, n_real=metrics.n_real, n_pos=metrics.n_pos,
                   status=metrics.status, runtime=metrics.runtime, seed=seed, idx=i)
        if i % 50 == 0
            @info "progress" i reward metrics.n_real metrics.n_pos metrics.status
        end
    end

    # Flatten to DataFrame
    df = DataFrame()
    for j in 1:D
        df[!, Symbol("a_$j")] = [rows[i].a[j] for i in 1:N]
    end
    df.reward   = [rows[i].reward for i in 1:N]
    df.n_real   = [rows[i].n_real for i in 1:N]
    df.n_pos    = [rows[i].n_pos  for i in 1:N]
    df.status   = [rows[i].status for i in 1:N]
    df.runtime  = [rows[i].runtime for i in 1:N]
    df.seed     = [rows[i].seed for i in 1:N]
    df.idx      = [rows[i].idx for i in 1:N]

    CSV.write(outpath, df)
    return df
end