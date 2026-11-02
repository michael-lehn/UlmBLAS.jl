#
#  julia --project=lapack lapack/test.jl: DGETRF of Netlib LAPACK on top of
#  ulmBLAS (and of OpenBLAS), compared with Julia's default (OpenBLAS).
#
#  For square, rectangular and singular matrices, matrices with a known
#  sequence of pivots and with a leading dimension larger than m:
#
#  - INFO and the pivots equal those of OpenBLAS (and the expected ones
#    where they are known), 1-based with i <= ipiv[i] <= m;
#  - residual ||P*A - L*U||_1 / (||A||_1 max(m,n) eps) <= 30 (the
#    threshold of LAPACK's tests);
#  - rows beyond m (leading dimension > m) stay unchanged;
#  - lu(A) and A \ b of LinearAlgebra (DGETRF and DGETRS through
#    libblastrampoline) agree with OpenBLAS.
#
include("common.jl")
using Random, Test, Printf

const THRESHOLD = 30

#  P*A = L*U with the interchanges ipiv: the rows of A in the order p
function permutation(ipiv, m)
    p = collect(1:m)
    for (i, j) in enumerate(ipiv)
        p[i], p[j] = p[j], p[i]
    end
    return p
end

function residual(A, F, ipiv)
    m, n = size(A); k = min(m, n)
    L = [i == j ? 1.0 : i > j ? F[i, j] : 0.0 for i in 1:m, j in 1:k]
    U = [i <= j ? F[i, j] : 0.0 for i in 1:k, j in 1:n]
    R = A[permutation(ipiv, m), :] - L*U
    nA = opnorm(A, 1)
    return nA == 0 ? opnorm(R, 1) : opnorm(R, 1)/(nA*max(m, n)*eps())
end

#  DGETRF of A (copied into an array with leading dimension m + pad)
function factor(A; pad=0)
    m, n = size(A)
    W = fill(-7.0, m + pad, n)
    W[1:m, :] .= A
    ipiv = fill(BlasInt(-1), min(m, n))
    info = dgetrf!(view(W, 1:m, :), ipiv)
    untouched = all(==(-7.0), W[m+1:end, :])
    return W[1:m, :], ipiv, info, untouched
end

#  A = P*L*U with |L[i,j]| <= 0.5: partial pivoting chooses the rows of P,
#  so the pivots are known
function known_pivots(rng, m, n)
    k = min(m, n)
    L = [i == j ? 1.0 : i > j ? rand(rng)-0.5 : 0.0 for i in 1:m, j in 1:k]
    U = [i <= j ? (i == j ? 2 + rand(rng) : randn(rng)) : 0.0 for i in 1:k, j in 1:n]
    p = randperm(rng, m)
    A = zeros(m, n)
    A[p, :] = L*U
    #  the interchanges that bring row p[i] to position i
    q = collect(1:m); pos = invperm(q)
    ipiv = zeros(BlasInt, k)
    for i in 1:k
        j = pos[p[i]]
        ipiv[i] = j
        q[i], q[j] = q[j], q[i]
        pos[q[i]], pos[q[j]] = i, j
    end
    return A, ipiv
end

function cases()
    rng = Xoshiro(20261102)
    c = Tuple{String,Matrix{Float64},Int,Union{Nothing,Vector{BlasInt}},Union{Nothing,Int}}[]
    for n in (1, 2, 3, 7, 31, 63, 64, 65, 127, 128, 129, 200, 500, 1000)
        push!(c, ("random $n×$n", randn(rng, n, n), 0, nothing, 0))
    end
    for (m, n) in ((300, 100), (100, 300), (1, 50), (50, 1), (129, 64), (64, 129),
                   (1000, 257), (257, 1000))
        push!(c, ("random $m×$n", randn(rng, m, n), 0, nothing, 0))
    end
    for (m, n) in ((100, 100), (200, 200), (65, 30), (30, 65))
        push!(c, ("random $m×$n, lda = $(m+13)", randn(rng, m, n), 13, nothing, 0))
    end
    for (m, n) in ((10, 10), (200, 200), (300, 120), (120, 300))
        A, ipiv = known_pivots(rng, m, n)
        push!(c, ("known pivots $m×$n", A, 0, ipiv, 0))
    end
    #  a pivot in the last row at every step
    n = 150
    A = [i + j == n + 1 ? 1.0 : 1e-3*randn(rng) for i in 1:n, j in 1:n]
    push!(c, ("anti-diagonal $n×$n", A, 0, nothing, 0))
    for n in (1, 5, 100)
        push!(c, ("zero $n×$n", zeros(n, n), 0, collect(BlasInt, 1:n), 1))
    end
    for n in (5, 100)
        push!(c, ("ones $n×$n", ones(n, n), 0, nothing, 2))
    end
    for (n, k) in ((8, 3), (100, 37), (300, 64), (300, 65), (300, 300))
        A = randn(rng, n, n); A[:, k] .= 0
        push!(c, ("zero column $k of $n×$n", A, 0, nothing, k))
    end
    A = randn(rng, 120, 80); A[:, 50] .= 0
    push!(c, ("zero column 50 of 120×80", A, 0, nothing, 50))
    return c
end

describe()
const CASES = cases()

#  the residuals are computed with OpenBLAS (the same for all configurations)
configure(:openblas)
const REF = [factor(A; pad) for (_, A, pad, _, _) in CASES]
const REF_RES = [residual(A, F, ipiv) for ((_, A), (F, ipiv)) in zip(CASES, REF)]
const REF_LU = [lu(A; check=false) for (_, A, _, _, _) in CASES]

@testset "DGETRF" begin
for c in (:netlib_ulmblas, :netlib_openblas)
    configure(c)
    config = string(BLAS.get_config())
    results = [factor(A; pad) for (_, A, pad, _, _) in CASES]
    #  LinearAlgebra: lu and the solution of A x = b (square, nonsingular)
    linalg = map(enumerate(CASES)) do (i, (_, A))
        size(A, 1) == size(A, 2) && REF[i][3] == 0 || return nothing
        Fl = lu(A)
        x = A \ randn(Xoshiro(i), size(A, 1))
        return Fl.p, Fl.L*Fl.U ≈ A[Fl.p, :], x
    end
    configure(:openblas)
    println("\n", c, ": ", config)
    @printf("  %-30s %6s %8s %10s %8s\n", "case", "INFO", "pivots", "residual", "OpenBLAS")
    @testset "$c" begin
        for (i, (name, A, pad, ipiv_expected, info_expected)) in enumerate(CASES)
            F, ipiv, info, untouched = results[i]
            _, ipivr, infor, _ = REF[i]
            r = residual(A, F, ipiv)
            m, n = size(A)
            pivots_ok = ipiv == ipivr
            @printf("  %-30s %6d %8s %10.3g %8.3g\n", name, info,
                    pivots_ok ? "= ref" : "DIFFER", r, REF_RES[i])
            @test info == infor
            info_expected === nothing || @test info == info_expected
            @test all(i <= ipiv[i] <= m for i in eachindex(ipiv))
            @test pivots_ok
            ipiv_expected === nothing || @test ipiv == ipiv_expected
            @test r <= THRESHOLD
            @test untouched
            if linalg[i] !== nothing
                p, lu_ok, x = linalg[i]
                b = randn(Xoshiro(i), n)
                @test p == REF_LU[i].p
                @test lu_ok
                @test norm(A*x - b, 1) <= THRESHOLD*opnorm(A, 1)*norm(x, 1)*n*eps()
            end
        end
    end
end
end
