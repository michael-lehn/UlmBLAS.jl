#
#  GEMM benchmark: ulmBLAS.mul! (ulmblas_dgemm_strided) against
#  LinearAlgebra.mul! with OpenBLAS, for a strided view (row stride 2) and
#  a contiguous matrix, n = 1000 and 1500, 1 to 4 BLAS threads.
#
#      cd bench
#      ULMBLAS_LIBRARY=/path/libulmblas_lbt.dylib julia --project=. gemm_threads.jl
#
#  `using ulmBLAS` forwards the BLAS routines of libblastrampoline (LBT) to
#  ulmBLAS.  For this benchmark LBT is switched back to OpenBLAS alone, so
#  that LinearAlgebra.mul! measures OpenBLAS; ulmBLAS.mul! calls
#  ulmblas_dgemm_strided through its own function pointer and does not go
#  through LBT.  Both are checked before measuring:
#
#  - the address LBT forwards dgemm_ to is the dgemm_64_ of OpenBLAS;
#  - a counting wrapper in LBT shows how many dgemm_ calls LinearAlgebra.mul!
#    makes (none for the strided view: generic Julia code instead of BLAS);
#  - the function pointer of ulmBLAS.mul! is ulmblas_dgemm_strided of the
#    ulmBLAS library;
#  - BLAS.set_num_threads sets the threads of OpenBLAS (read back from
#    OpenBLAS); ulmBLAS is single-threaded, the generic Julia code as well.
#
using BenchmarkTools, LinearAlgebra, Libdl, Printf, Random
if !haskey(ENV, "ULMBLAS_LIBRARY")
    lib = joinpath(@__DIR__, "..", "dev", "lib",
                   Sys.isapple() ? "libulmblas_lbt.dylib" : "libulmblas_lbt.so")
    isfile(lib) && (ENV["ULMBLAS_LIBRARY"] = abspath(lib))
end
using ulmBLAS
import OpenBLAS_jll

const ulmlib = ulmBLAS.library_path()
const oblib  = OpenBLAS_jll.libopenblas_path

symaddr(lib, name) = UInt(dlsym(dlopen(lib), name))
forward(name) = UInt(BLAS.lbt_get_forward(name, :ilp64))

function owner(name)
    a = forward(name)
    a == symaddr(oblib, name * "64_") ? "OpenBLAS" :
    a == symaddr(ulmlib, name * "64_") ? "ulmBLAS" : "other"
end

println("library of ulmBLAS: ", ulmlib)
println("LBT after `using ulmBLAS`: dgemm_ -> ", owner("dgemm_"))

#  LBT: OpenBLAS only
BLAS.lbt_forward(oblib; clear=true, verbose=false, suffix_hint="64_")
@assert owner("dgemm_") == "OpenBLAS"
println("LBT for this benchmark:   dgemm_ -> ", owner("dgemm_"),
        " (", basename(oblib), ")")

const strided = ulmBLAS.gemm_strided_ptr[Float64]
@assert UInt(strided) == symaddr(ulmlib, "ulmblas_dgemm_strided")
println("ulmBLAS.mul!:             ulmblas_dgemm_strided of ", basename(ulmlib))

#  Counting wrapper around the dgemm_ that LBT forwards to
const realgemm = BLAS.lbt_get_forward("dgemm_", :ilp64)
const ncalls = Ref(0)
function countgemm(ta::Ptr{UInt8}, tb::Ptr{UInt8}, m::Ptr{Int64},
                   n::Ptr{Int64}, k::Ptr{Int64}, al::Ptr{Float64},
                   A::Ptr{Float64}, lda::Ptr{Int64}, B::Ptr{Float64},
                   ldb::Ptr{Int64}, be::Ptr{Float64}, C::Ptr{Float64},
                   ldc::Ptr{Int64}, l1::Csize_t, l2::Csize_t)
    ncalls[] += 1
    ccall(realgemm, Cvoid,
          (Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
           Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64},
           Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Csize_t, Csize_t),
          ta, tb, m, n, k, al, A, lda, B, ldb, be, C, ldc, l1, l2)
end
function gemmcalls(f)
    cf = @cfunction(countgemm, Cvoid,
                    (Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64},
                     Ptr{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int64},
                     Ptr{Float64}, Ptr{Int64}, Ptr{Float64}, Ptr{Float64},
                     Ptr{Int64}, Csize_t, Csize_t))
    BLAS.lbt_set_forward("dgemm_", cf, :ilp64)
    ncalls[] = 0
    try
        f()
    finally
        BLAS.lbt_set_forward("dgemm_", realgemm, :ilp64)
    end
    ncalls[]
end
let P = rand(64, 32), A = view(P, 1:2:64, :), Ac = Matrix(A), B = rand(32, 32),
    C = zeros(32, 32)
    s = gemmcalls(() -> LinearAlgebra.mul!(C, A, B))
    c = gemmcalls(() -> LinearAlgebra.mul!(C, Ac, B))
    u = gemmcalls(() -> ulmBLAS.mul!(C, A, B))
    println("dgemm_ calls through LBT: LinearAlgebra.mul! strided ", s,
            ", contiguous ", c, "; ulmBLAS.mul! ", u)
    @assert s == 0 && c == 1 && u == 0
end
@assert owner("dgemm_") == "OpenBLAS"

obthreads() = ccall((:openblas_get_num_threads64_, oblib), Cint, ())
cpu = Sys.isapple() ? readchomp(`sysctl -n machdep.cpu.brand_string`) :
                      Sys.cpu_info()[1].model
cores = Sys.isapple() ?
    string(readchomp(`sysctl -n hw.perflevel0.physicalcpu`), " performance + ",
           readchomp(`sysctl -n hw.perflevel1.physicalcpu`), " efficiency cores") :
    string(Sys.CPU_THREADS, " threads")
println("CPU: ", cpu, " (", cores, "), Julia ", VERSION, ", Julia threads ",
        Threads.nthreads(), ", OpenBLAS ", BLAS.get_config().loaded_libs[1].libname |> basename)

#  Benchmarks
const cases = (("ulmBLAS.mul!, strided",          :ulm, true),
               ("LinearAlgebra.mul!, strided",    :jl,  true),
               ("LinearAlgebra.mul!, contiguous", :jl,  false),
               ("ulmBLAS.mul!, contiguous",       :ulm, false))
backend(kind, strided) = kind === :ulm ? "ulmBLAS (1 thread)" :
                         strided ? "generic Julia (1 thread)" : "OpenBLAS"
results = []
println()
@printf("%-31s %-5s %-24s %8s %7s %9s %7s %6s\n", "case", "n",
        "backend", "threads", "ms", "GFLOPS", "bytes", "allocs")
for t in 1:4
    BLAS.set_num_threads(t)
    @assert BLAS.get_num_threads() == t && obthreads() == t
    for n in (1000, 1500)
        Random.seed!(n)
        P  = rand(2n, n)
        A  = view(P, 1:2:2n, :)
        Ac = Matrix(A)
        B  = rand(n, n)
        C  = zeros(n, n)
        R  = Ac * B                                    # OpenBLAS
        for (name, kind, str) in cases
            X = str ? A : Ac
            fill!(C, NaN)
            kind === :ulm ? ulmBLAS.mul!(C, X, B) : LinearAlgebra.mul!(C, X, B)
            @assert C ≈ R "$name, n = $n: wrong result"
            b = kind === :ulm ? @benchmark(ulmBLAS.mul!($C, $X, $B)) :
                                @benchmark(LinearAlgebra.mul!($C, $X, $B))
            tmin = minimum(b).time
            th = kind === :jl && !str ? string(t) : "1"
            @printf("%-31s %-5d %-24s %8s %7.1f %9.1f %7d %6d\n", name, n,
                    backend(kind, str), th, tmin/1e6, 2.0*n^3/tmin,
                    b.memory, b.allocs)
            push!(results, (name, n, backend(kind, str), t, tmin, median(b).time,
                            b.memory, b.allocs, length(b.times)))
        end
    end
end

open(joinpath(@__DIR__, "gemm_threads.tsv"), "w") do io
    println(io, "case\tn\tbackend\tBLAS_threads\tmin_ns\tmedian_ns\tbytes\tallocs\tsamples")
    for r in results
        println(io, join(r, '\t'))
    end
end
