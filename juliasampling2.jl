# wnt_sampling.jl
using HomotopyContinuation
using Random
using DataFrames
using CSV
using Dates

@var x[1:19]
@var k[1:31]
@var c[1:5]

# --- Steady-state equations (14 generators) ---
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

# --- Conservation laws (5) ---
g1 = (x[1] + x[2] + x[3] + x[14] + x[15]) - c[1]
g2 = (x[4] + x[5] + x[6] + x[7] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19]) - c[2]
g3 = (x[8] + x[16]) - c[3]
g4 = (x[9] + x[17]) - c[4]
g5 = (x[12] + x[13]) - c[5]

polys = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f14,f18,g1,g2,g3,g4,g5]
params = vcat(k, c)

# helper: make a concrete system at parameter point
function concrete_system(polys, params, parvals, x)
    pdict = Dict(params[i] => parvals[i] for i in eachindex(params))
    polysθ = subs.(polys, Ref(pdict))           # substitute numbers into each polynomial
    return System(polysθ; variables=x)
end

function run_experiment(; N=10_000, seed=1, tol_real=1e-7, out_csv="wnt_samples.csv")
    rng = MersenneTwister(seed)

    cols = Dict{Symbol, Any}()
    cols[:trial] = Int[]
    cols[:ok] = Bool[]
    cols[:n_solutions] = Int[]
    cols[:n_real] = Int[]
    for i in 1:31; cols[Symbol("k$i")] = Float64[]; end
    for i in 1:5;  cols[Symbol("c$i")] = Float64[]; end
    cols[:err] = String[]
    df = DataFrame(cols)

    println("Starting N=$N trials at $(now()). Output: $out_csv")

    for t in 1:N
        θ = -1 .+ 2 .* rand(rng, 36)    # uniform in [-1,1]
        kvals = θ[1:31]
        cvals = θ[32:36]
        parvals = vcat(kvals, cvals)

        ok = true
        nsol = 0
        nreal = 0
        err = ""

        try
            Fθ = concrete_system(polys, params, parvals, x)
            res = solve(Fθ; start_system=:polyhedral)

            sols = solutions(res)
            nsol = length(sols)
            nreal = length(real_solutions(res; only_finite=true))

        catch e
            ok = false
            bt = catch_backtrace()
            err = sprint(showerror, e, bt)   # <-- this will NOT be empty
            if t ≤ 3
                println("\nFirst failure at trial $t:\n$err\n")
            end
        end

        push!(df.trial, t)
        push!(df.ok, ok)
        push!(df.n_solutions, nsol)
        push!(df.n_real, nreal)
        for i in 1:31; push!(df[!, Symbol("k$i")], kvals[i]); end
        for i in 1:5;  push!(df[!, Symbol("c$i")], cvals[i]); end
        push!(df.err, err)

        if t % 100 == 0
            println("trial $t / $N | ok=$(count(df.ok)) | last: nsol=$nsol, nreal=$nreal")
        end
    end

    CSV.write(out_csv, df)
    println("Done. Wrote $(nrow(df)) rows to $out_csv")
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment(N=10_000, seed=1, tol_real=1e-7, out_csv="wnt_samples_robust.csv")
end