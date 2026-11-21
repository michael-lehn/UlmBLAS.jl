#
#  Cholesky factorization (LAPACK.potrf!, as in cholesky!) with Julia's
#  default BLAS/LAPACK (OpenBLAS) against ulmBLAS (after UlmBLAS.forward()),
#  for n = 100 ... 4000 and two thread numbers.
#
#      cd bench
#      julia --project=. potrf.jl
#
#  THREADS="1 6" sets the thread numbers (default: 1 and Julia's default
#  number of BLAS threads), SIZES="100 200 ..." the sizes, RESULTS=name
#  writes name.tsv instead of potrf.tsv.
#
#  OpenBLAS is measured first, then UlmBLAS.forward() and ulmBLAS.  Before
#  each part the script checks where libblastrampoline (LBT) forwards
#  dpotrf_ to.  Each factorization is checked (residual of L Lᵀ = A).
#  Time: median of BenchmarkTools samples, each on a fresh copy of A.
#  GFLOPS: n³/3 floating point operations per time.
#
using BenchmarkTools, LinearAlgebra, Libdl, Printf, Random
using UlmBLAS
import OpenBLAS_jll

const oblib = OpenBLAS_jll.libopenblas_path

symaddr(lib, name) = UInt(dlsym(dlopen(lib), name; throw_error=false))
forward(name) = UInt(BLAS.lbt_get_forward(name, :ilp64))

function owner(name)
    a = forward(name)
    a == symaddr(oblib, name * "64_") ? "OpenBLAS" :
    a == symaddr(UlmBLAS.library_path(), name * "64_") ? "ulmBLAS" : "other"
end

parseints(s) = parse.(Int, split(s))
const sizes = parseints(get(ENV, "SIZES", "100 200 500 1000 2000 4000"))
const threads = haskey(ENV, "THREADS") ? parseints(ENV["THREADS"]) :
                unique([1, BLAS.get_num_threads()])
const results = get(ENV, "RESULTS", "potrf") * ".tsv"

function spd(n)
    Random.seed!(n)
    X = randn(n, n)
    A = X * X'
    A + n * I
end

function residual(A, L)
    n = size(A, 1)
    T = LowerTriangular(L)
    norm(T * T' - Symmetric(A, :L)) / (n * eps() * norm(A))
end

function measure(A)
    B = copy(A)
    LAPACK.potrf!('L', B)
    r = residual(A, B)
    t = @belapsed LAPACK.potrf!('L', B) setup = (B = copy($A)) evals = 1 seconds = 3
    t, r
end

cpu = Sys.cpu_info()[1].model
println("CPU: ", cpu, ", ", Sys.CPU_THREADS, " hardware threads")
println("Julia ", VERSION, ", ", basename(oblib), ", ulmBLAS: ",
        basename(UlmBLAS.library_path()))

rows = []
for (lib, setup) in (("OpenBLAS", () -> nothing), ("ulmBLAS", UlmBLAS.forward))
    setup()
    @assert owner("dpotrf_") == lib "dpotrf_ goes to $(owner("dpotrf_"))"
    println("\n", lib, " (dpotrf_ -> ", owner("dpotrf_"), ")")
    for p in threads
        BLAS.set_num_threads(p)
        for n in sizes
            A = spd(n)
            t, r = measure(A)
            r < 100 || error("$lib n = $n: residual $r")
            g = n^3 / 3 / t / 1e9
            @printf("  threads %2d  n %5d  %10.3f ms  %7.1f GFLOPS  (residual %.2f)\n",
                    p, n, 1e3 * t, g, r)
            push!(rows, (lib, p, n, t, g))
        end
    end
end

open(results, "w") do io
    println(io, "library\tthreads\tn\tseconds\tgflops")
    for (lib, p, n, t, g) in rows
        println(io, lib, '\t', p, '\t', n, '\t', t, '\t', g)
    end
end

println("\nulmBLAS / OpenBLAS (time of OpenBLAS / time of ulmBLAS):")
for p in threads, n in sizes
    to = only(r[4] for r in rows if r[1] == "OpenBLAS" && r[2] == p && r[3] == n)
    tu = only(r[4] for r in rows if r[1] == "ulmBLAS" && r[2] == p && r[3] == n)
    @printf("  threads %2d  n %5d  %5.2f\n", p, n, to / tu)
end
