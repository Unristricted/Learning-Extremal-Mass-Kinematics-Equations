# biotest.jl
# Wnt shuttle model (arXiv:1502.03188) region-scan harness for HomotopyContinuation.jl
#
# Fixes vs your crashing version:
#   (A) Pass parameters as a VECTOR aligned with `params = [k...; c...]`
#   (B) Force start_system = :total_degree to avoid polyhedral recursion/stack overflow
#
# Usage:
#   julia --project -e 'include("biotest.jl"); scan_regions()'
#   julia --project -e 'include("biotest.jl"); probe_C(-0.6)'

using HomotopyContinuation
using LinearAlgebra
using Random

# -----------------------------
# 1) Symbolic variables
# -----------------------------
@var x[1:19]              # species
@var k[1:31]              # rate constants
@var c[1:5]               # conserved quantities

# -----------------------------
# 2) ODE RHS = 0 (Eq. (1) plus implied equalities)
# -----------------------------
f1  = -k[1]*x[1] + k[2]*x[2]
f2  =  k[1]*x[1] - (k[2] + k[26])*x[2] + k[27]*x[3] - k[3]*x[2]*x[4] + (k[4] + k[5])*x[14]
f3  =  k[26]*x[2] - k[27]*x[3] - k[14]*x[3]*x[6] + (k[15] + k[16])*x[15]
f4  = -k[3]*x[2]*x[4] - k[9]*x[4]*x[10] + k[4]*x[14] + k[8]*x[16] + (k[10] + k[11])*x[18]
f5  = -k[28]*x[5] + k[29]*x[7] - k[6]*x[5]*x[8] + k[5]*x[14] + k[7]*x[16]
f6  = -k[14]*x[3]*x[6] - k[20]*x[6]*x[11] + k[15]*x[15] + k[19]*x[17] + (k[21] + k[22])*x[19]
f7  =  k[28]*x[5] - k[29]*x[7] - k[17]*x[7]*x[9] + k[16]*x[15] + k[18]*x[17]
f8  = -k[6]*x[5]*x[8] + (k[7] + k[8])*x[16]                     # x8dot
f9  = -k[17]*x[7]*x[9] + (k[18] + k[19])*x[17]                  # x9dot
f10 =  k[12] - (k[13] + k[30])*x[10] - k[9]*x[4]*x[10] + k[31]*x[11] + k[10]*x[18]
f11 = -k[23]*x[11] + k[30]*x[10] - k[31]*x[11] - k[20]*x[6]*x[11] - k[24]*x[11]*x[12] + k[25]*x[13] + k[21]*x[19]
f12 = -k[24]*x[11]*x[12] + k[25]*x[13]                          # x12dot
f14 =  k[3]*x[2]*x[4] - (k[4] + k[5])*x[14]
f15 =  k[14]*x[3]*x[6] - (k[15] + k[16])*x[15]
f18 =  k[9]*x[4]*x[10] - (k[10] + k[11])*x[18]
f19 =  k[20]*x[6]*x[11] - (k[21] + k[22])*x[19]

# implied equalities:
f13 = -f12
f16 = -f8
f17 = -f9

ode_eqs = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19]

# -----------------------------
# 3) Conservation laws (Eq. (2))
# -----------------------------
g1 = (x[1] + x[2] + x[3] + x[14] + x[15]) - c[1]
g2 = (x[4] + x[5] + x[6] + x[7] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19]) - c[2]
g3 = (x[8] + x[16])  - c[3]
g4 = (x[9] + x[17])  - c[4]
g5 = (x[12] + x[13]) - c[5]

cons_eqs = [g1,g2,g3,g4,g5]

# Overdetermined system: 24 equations in 19 unknowns
F_over = vcat(ode_eqs, cons_eqs)

# IMPORTANT: parameter order is EXACTLY this vector
params = vcat(k..., c...)

sys_over = System(F_over; variables=x, parameters=params)

# -----------------------------
# 4) Example 4.3 parameter slice
# -----------------------------
function param_vector_for_C(C::Real)::Vector{Float64}
    # k as in Example 4.3
    kvals = [
        9/5, 9/5, 3, 2/3, 2/3, 3, 1, 1, 100, 4/5, 80, 100, 1, 3, 2/3, 2/3, 38,
        4/5, 4/5, 4, 1/8, 3/5, 1, 1/2, 19, 7/4, 7/4, 1, 1, 5, 1
    ]
    # c = c(C)
    cvals = [5, 16 + C, 8/5 - C, 65 + C, 3 - C]
    return Float64.(vcat(kvals, cvals))  # length 36, aligned with params
end

# -----------------------------
# 5) Discriminant breakpoints (Example 4.3)
# -----------------------------
Cbreaks = [
    -77.2388, -16.0, -5.28669, -1.57472, -1.46506, -1.34899, -1.29581, -1.2,
    -1.19215, -1.18389, -0.584325, -0.361808, 0.191039, 1.30812, 1.33197, 1.6,
    1.60161, 3.0, 4.26306, 11.1174, 21.4165, 310.141
]
sort!(Cbreaks)

# -----------------------------
# 6) Solve + count positive real solutions
# -----------------------------
function count_positive_real_solutions(sys::System, pvec::Vector{Float64};
                                      real_tol=1e-8, pos_tol=1e-10,
                                      start_system::Symbol=:total_degree)
    # KEY: start_system=:total_degree avoids the polyhedral recursion you hit
    res = solve(sys; target_parameters=pvec, start_system=start_system)

    reals = real_solutions(res; tol=real_tol)
    npos = count(sol -> all(sol .> pos_tol), reals)

    return (npos=npos, nreal=length(reals), result=res, reals=reals)
end

# -----------------------------
# 7) Region scan
# -----------------------------
function scan_regions(; margin=1e-3,
                       real_tol=1e-8, pos_tol=1e-10,
                       outer_left=-1e3, outer_right=1e3,
                       start_system::Symbol=:total_degree)

    intervals = Vector{Tuple{Float64,Float64}}()
    push!(intervals, (Float64(outer_left), Float64(Cbreaks[1])))
    for i in 1:length(Cbreaks)-1
        push!(intervals, (Float64(Cbreaks[i]), Float64(Cbreaks[i+1])))
    end
    push!(intervals, (Float64(Cbreaks[end]), Float64(outer_right)))

    for (a,b) in intervals
        mid = (a + b)/2
        Ctest = clamp(mid, a + margin, b - margin)

        println("\n--- Region (", a, ", ", b, ")  test at C = ", Ctest, " ---")

        pvec = param_vector_for_C(Ctest)
        out = count_positive_real_solutions(sys_over, pvec;
                                            real_tol=real_tol,
                                            pos_tol=pos_tol,
                                            start_system=start_system)
        println("C = $Ctest  =>  real solutions = $(out.nreal), positive real = $(out.npos)")
    end
    return nothing
end

# -----------------------------
# 8) Single-point probe
# -----------------------------
function probe_C(C::Real; real_tol=1e-8, pos_tol=1e-10, start_system::Symbol=:total_degree)
    println("\n--- Probe at C = $C ---")
    pvec = param_vector_for_C(C)
    out = count_positive_real_solutions(sys_over, pvec;
                                        real_tol=real_tol,
                                        pos_tol=pos_tol,
                                        start_system=start_system)
    println("C = $C  =>  real solutions = $(out.nreal), positive real = $(out.npos)")
    return out
end

# Uncomment for quick local test:
scan_regions()
probe_C(-0.6)