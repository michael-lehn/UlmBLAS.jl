#
#  julia --project=lapack lapack/bench.jl OUT.tsv [n ...]: time of a complete
#  DGETRF (n×n, called through libblastrampoline as LinearAlgebra.LAPACK
#  does) in each configuration of common.jl, single-threaded.
#
#  MATRIX (default "dense"): comma-separated kinds of test_matrix in
#  common.jl, e.g. MATRIX=dense,coldom.  For every n the same matrices
#  (seed n) are factored in all configurations; the kinds and the
#  configurations take turns (the order of the configurations rotates with
#  n).  A sample is one call, after copying the matrix into the work array
#  (not timed); samples are taken for BUDGET seconds (default 1) per n,
#  kind and configuration, at least 5 and at most 2000, after 2 calls for
#  warm-up.  The pivots are compared with those of OpenBLAS; for coldom
#  they must be ipiv[i] == i (no row interchanges), otherwise bench.jl
#  stops.
#
#  Columns: n, matrix, config, samples, median_ns, min_ns, q25_ns, q75_ns,
#  gflops (2n^3/3 / median), ipiv_equal (pivots equal to OpenBLAS),
#  interchanges (number of i with ipiv[i] != i).
#
include("common.jl")
using Random, Statistics, Printf

const SIZES = length(ARGS) >= 2 ? parse.(Int, ARGS[2:end]) :
    [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
     1536, 2048, 3072, 4096]
const BUDGET = parse(Float64, get(ENV, "BUDGET", "1.0"))
out = ARGS[1]

function samples(A, B, ipiv)
    for _ in 1:2
        copyto!(B, A); dgetrf!(B, ipiv) == 0 || error("singular")
    end
    t = Float64[]
    start = time()
    while length(t) < 5 || (time() - start < BUDGET && length(t) < 2000)
        copyto!(B, A)
        t0 = time_ns()
        dgetrf!(B, ipiv)
        push!(t, time_ns() - t0)
    end
    return t
end

describe()
configure(:openblas)
println("BLAS threads: ", BLAS.get_num_threads(), ", Julia threads: ", Threads.nthreads())
const MATRICES = Symbol.(split(get(ENV, "MATRIX", "dense"), ','))
open(out, "w") do io
    println(io, join(["n", "matrix", "config", "samples", "median_ns", "min_ns",
                      "q25_ns", "q75_ns", "gflops", "ipiv_equal", "interchanges"], '\t'))
    for (k, n) in enumerate(SIZES), kind in MATRICES
        A = test_matrix(kind, n); B = similar(A)
        configure(:openblas)
        ipiv_ref = zeros(BlasInt, n); copyto!(B, A); dgetrf!(B, ipiv_ref)
        order = circshift(collect(CONFIGS), k - 1)
        for c in order
            configure(c)
            ipiv = zeros(BlasInt, n)
            t = samples(A, B, ipiv)
            swaps = count(i -> ipiv[i] != i, 1:n)
            kind === :coldom && swaps > 0 && error("coldom, n = $n, $c: $swaps row interchanges")
            med = median(t)
            gf = (2n^3/3)/med
            q25, q75 = quantile(t, (0.25, 0.75))
            println(io, join([n, kind, c, length(t), med, minimum(t), q25, q75,
                              round(gf, digits=3), ipiv == ipiv_ref, swaps], '\t'))
            flush(io)
            @printf("n = %5d  %-8s %-16s %5d samples  median %12.0f ns  %7.2f GFLOPS  %5d interchanges%s\n",
                    n, kind, c, length(t), med, gf, swaps, ipiv == ipiv_ref ? "" : "  (pivots differ)")
        end
    end
end
configure(:openblas)
