# wnt_oracle.jl
module WNTOracle

using Pkg
Pkg.add(["HomotopyContinuation", "DynamicPolynomials"])

using HomotopyContinuation
using DynamicPolynomials

export evaluate_params

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

const polys  = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f14,f18,g1,g2,g3,g4,g5]
const params = vcat(k, c)

function concrete_system(parvals::AbstractVector{<:Real})
    pdict  = Dict(params[i] => parvals[i] for i in eachindex(params))
    polysθ = subs.(polys, Ref(pdict))
    return System(polysθ; variables=x)
end

function count_real_and_pos(res; tol_real=1e-7, tol_pos=1e-9)
    nreal = 0
    npos  = 0
    for s in solutions(res)
        # `solutions(res)` may yield solution objects with field `u`
        # (e.g. `Root`-like structs) or may directly yield arrays.
        # Handle both cases: prefer `s.u` when available, otherwise use `s`.
        z = hasproperty(s, :u) ? s.u : s
        if maximum(abs.(imag.(z))) <= tol_real
            nreal += 1
            if minimum(real.(z)) >= tol_pos
                npos += 1
            end
        end
    end
    return nreal, npos
end

"""
evaluate_params(parvals; start_system=:polyhedral, tol_real=1e-7, tol_pos=1e-9)

parvals is length-36: [k1..k31, c1..c5]
Returns NamedTuple: (ok, n_solutions, n_real, n_pos, err)
"""
function evaluate_params(parvals::AbstractVector{<:Real};
                         start_system::Symbol = :polyhedral,
                         tol_real::Float64 = 1e-7,
                         tol_pos::Float64 = 1e-9)

    ok = true
    nsol = 0
    nreal = 0
    npos = 0
    err = ""

    try
        Fθ = concrete_system(parvals)
        res = solve(Fθ; start_system=start_system)
        nsol = length(solutions(res))
        nreal, npos = count_real_and_pos(res; tol_real=tol_real, tol_pos=tol_pos)
    catch e
        ok = false
        err = sprint(showerror, e, catch_backtrace())
    end

    return Dict(
    "ok" => ok,
    "n_solutions" => nsol,
    "n_real" => nreal,
    "n_pos" => npos,
    "err" => err
    )
end

end # module