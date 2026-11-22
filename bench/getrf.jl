#
#  LU factorization with partial pivoting (LAPACK.getrf!, as in lu!) with
#  Julia's default BLAS/LAPACK (OpenBLAS) against ulmBLAS (after
#  UlmBLAS.forward()), for n = 100 ... 4000 and two thread numbers.
#
#      cd bench
#      julia --project=. getrf.jl
#
#  THREADS="1 6" sets the thread numbers (default: 1 and Julia's default
#  number of BLAS threads), SIZES="100 200 ..." the sizes, RESULTS=name
#  writes name.tsv instead of getrf.tsv.
#
#  OpenBLAS is measured first, then UlmBLAS.forward() and ulmBLAS.  Before
#  each part the script checks where libblastrampoline (LBT) forwards
#  dgetrf_ to.  Each factorization is checked (residual of P A = L U).
#  Time: median of BenchmarkTools samples, each on a fresh copy of A.
#  GFLOPS: 2n³/3 floating point operations per time.
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
const results = get(ENV, "RESULTS", "getrf") * ".tsv"

function matrix(n)
    Random.seed!(n)
    randn(n, n)
end

function residual(A, LU, ipiv)
    n = size(A, 1)
    F = LinearAlgebra.LU(LU, ipiv, 0)
    norm(F.L * F.U - A[F.p, :]) / (n * eps() * norm(A))
end

function measure(A)
    B = copy(A)
    _, ipiv, info = LAPACK.getrf!(B)
    info == 0 || error("getrf!: info = $info")
    r = residual(A, B, ipiv)
    t = @belapsed LAPACK.getrf!(B) setup = (B = copy($A)) evals = 1 seconds = 3
    t, r
end

cpu = Sys.cpu_info()[1].model
println("CPU: ", cpu, ", ", Sys.CPU_THREADS, " hardware threads")
println("Julia ", VERSION, ", ", basename(oblib), ", ulmBLAS: ",
        basename(UlmBLAS.library_path()))

rows = []
for (lib, setup) in (("OpenBLAS", () -> nothing), ("ulmBLAS", UlmBLAS.forward))
    setup()
    @assert owner("dgetrf_") == lib "dgetrf_ goes to $(owner("dgetrf_"))"
    println("\n", lib, " (dgetrf_ -> ", owner("dgetrf_"), ")")
    for p in threads
        BLAS.set_num_threads(p)
        for n in sizes
            A = matrix(n)
            t, r = measure(A)
            r < 100 || error("$lib n = $n: residual $r")
            g = 2 * n^3 / 3 / t / 1e9
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
