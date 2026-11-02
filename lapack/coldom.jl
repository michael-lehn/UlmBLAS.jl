#
#  julia --project=lapack lapack/coldom.jl [n ...]: the matrices of the
#  benchmark of matrices strictly diagonally dominant by columns
#  (test_matrix(:coldom, n) in common.jl, the sizes of bench.jl) in all three configurations:
#
#  - INFO = 0 and ipiv[i] == i for every i (no row interchanges);
#  - residual ||A - L*U||_1 / (||A||_1 n eps) <= 30 (computed with OpenBLAS);
#  - the pivots equal those of OpenBLAS.
#
#  For comparison, the number of row interchanges for the dense matrices
#  (randn) and for matrices dominant by rows only (test_matrix(:rowdom, n);
#  with random entries these happen to need none), and the 2×2 matrix
#  [2 1; 3 4], dominant by rows, but not by columns: one interchange.
#
include("common.jl")
using Test, Printf

const SIZES = isempty(ARGS) ?
    [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
     1536, 2048, 3072, 4096] : parse.(Int, ARGS)

function residual(A, F)
    L = UnitLowerTriangular(F); U = UpperTriangular(F)
    return opnorm(A - L*U, 1)/(opnorm(A, 1)*size(A, 1)*eps())
end

function factor(A)
    F = copy(A); ipiv = zeros(BlasInt, size(A, 1))
    info = dgetrf!(F, ipiv)
    return F, ipiv, info
end

swaps(ipiv) = count(i -> ipiv[i] != i, eachindex(ipiv))

describe()
@printf("\n%6s  %-16s %5s %12s %10s %8s   %s\n", "n", "config", "INFO",
        "interchanges", "residual", "= ref", "interchanges: dense, rowdom (OpenBLAS)")
@testset "column dominant" begin
    for c in CONFIGS
        configure(c)
        _, ipiv, info = factor([2.0 1.0; 3.0 4.0])
        println("[2 1; 3 4], $c: ipiv = $ipiv")
        @test info == 0 && ipiv == [2, 2]
    end
    for n in SIZES
        A = test_matrix(:coldom, n)
        configure(:openblas)
        others = [swaps(factor(test_matrix(k, n))[2]) for k in (:dense, :rowdom)]
        ref = factor(A)[2]
        for c in CONFIGS
            configure(c)
            F, ipiv, info = factor(A)
            configure(:openblas)
            r = residual(A, F)
            @printf("%6d  %-16s %5d %12d %10.3g %8s   %s\n", n, c, info, swaps(ipiv), r,
                    ipiv == ref, c === :openblas ? join(others, ", ") : "")
            @test info == 0
            @test ipiv == 1:n
            @test r <= 30
            @test ipiv == ref
        end
    end
end
configure(:openblas)
