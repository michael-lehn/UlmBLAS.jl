#
#  GEMM benchmark as gemm_threads.jl, with BLIS instead of OpenBLAS behind
#  LinearAlgebra.mul!: ulmBLAS.mul! (ulmblas_dgemm_strided) and
#  LinearAlgebra.mul! (BLIS) for a strided view (row stride 2) and a
#  contiguous matrix, n = 1000 and 1500, 1 to 4 BLIS threads.
#
#      cd bench
#      BLIS_LIBRARY=/path/libblis.dylib julia --project=. gemm_blis.jl
#
#  BLIS must have a 64-bit BLAS interface (configure -b 64) and pthreads or
#  OpenMP.  BLIS_SUFFIX is the suffix of its BLAS symbols (default none:
#  dgemm_).  BLIS_LIBRARY=blis_jll uses the BLIS of BLISBLAS.jl (symbols
#  dgemm_64_); blis_jll must then be in the load path, for example with
#  JULIA_LOAD_PATH="@:/path/env-with-blis_jll:@stdlib".  The number
#  of BLIS threads is set with bli_thread_set_num_threads and read back with
#  bli_thread_get_num_threads; the output also reports the observed
#  parallelism (process CPU time / wall time) for every case.
#
#  Before measuring it checks that LBT forwards dgemm_ to dgemm_ of BLIS,
#  counts the dgemm_ calls of LinearAlgebra.mul! (none for the strided view:
#  generic Julia code), checks that ulmBLAS.mul! calls ulmblas_dgemm_strided
#  of ulmBLAS, and compares the results with the generic Julia
#  multiplication.  RESULTS=name writes the results to name.tsv.
#
using BenchmarkTools, LinearAlgebra, Libdl, Printf, Random
if !haskey(ENV, "ULMBLAS_LIBRARY")
    lib = joinpath(@__DIR__, "..", "dev", "lib",
                   Sys.isapple() ? "libulmblas_lbt.dylib" : "libulmblas_lbt.so")
    isfile(lib) && (ENV["ULMBLAS_LIBRARY"] = abspath(lib))
end
using ulmBLAS

if ENV["BLIS_LIBRARY"] == "blis_jll"
    using blis_jll                      # also loads libomp (LLVMOpenMP_jll)
    const blislib = blis_jll.blis
    const blissuf = "64_"
else
    const blislib = abspath(ENV["BLIS_LIBRARY"])
    const blissuf = get(ENV, "BLIS_SUFFIX", "")
end
const ulmlib = ulmBLAS.library_path()

symaddr(lib, name) = UInt(dlsym(dlopen(lib), name))
forward(name) = UInt(BLAS.lbt_get_forward(name, :ilp64))

function owner(name)
    a = forward(name)
    a == symaddr(blislib, name * blissuf) ? "BLIS" :
    a == symaddr(ulmlib, name * "64_") ? "ulmBLAS" : "other"
end

println("library of ulmBLAS: ", ulmlib)
println("LBT after `using ulmBLAS`: dgemm_ -> ", owner("dgemm_"))

#  LBT: BLIS only (64-bit BLAS interface)
BLAS.lbt_forward(blislib; clear=true, verbose=false, suffix_hint=blissuf)
@assert owner("dgemm_") == "BLIS"
@assert BLAS.get_config().loaded_libs[1].interface === :ilp64
println("LBT for this benchmark:   dgemm_ -> ", owner("dgemm_"), " (dgemm_",
        blissuf, " of ", basename(blislib), ", ILP64)")

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
@assert owner("dgemm_") == "BLIS"

#  BLIS: version, configuration, threading
const hblis = dlopen(blislib)
bstr(f) = unsafe_string(ccall(dlsym(hblis, f), Cstring, ()))
bint(f) = ccall(dlsym(hblis, f), Int, ())
const arch = unsafe_string(ccall(dlsym(hblis, :bli_arch_string), Cstring, (Cint,),
                                 ccall(dlsym(hblis, :bli_arch_query_id), Cint, ())))
println("BLIS ", bstr(:bli_info_get_version_str), ": configuration ", arch,
        ", BLAS integer ", bint(:bli_info_get_blas_int_type_size), " bit",
        ", pthreads ", bint(:bli_info_get_enable_pthreads) != 0,
        ", OpenMP ", bint(:bli_info_get_enable_openmp) != 0)
setthreads(t) = ccall(dlsym(hblis, :bli_thread_set_num_threads), Cvoid, (Int,), t)
getthreads() = ccall(dlsym(hblis, :bli_thread_get_num_threads), Int, ())

cpu = readchomp(`sysctl -n machdep.cpu.brand_string`)
cores = string(readchomp(`sysctl -n hw.perflevel0.physicalcpu`), " performance + ",
               readchomp(`sysctl -n hw.perflevel1.physicalcpu`), " efficiency cores")
macos = string(readchomp(`sw_vers -productVersion`), " (", readchomp(`sw_vers -buildVersion`), ")")
println("CPU: ", cpu, " (", cores, "), macOS ", macos, ", Julia ", VERSION,
        ", Julia threads ", Threads.nthreads())

#  observed parallelism: process CPU time / wall time over repeated calls
function cpu_per_wall(f, tmin_ns)
    k = max(3, ceil(Int, 0.5e9/tmin_ns))
    f()
    t0 = time_ns(); c0 = ccall(:clock, Culong, ())
    for _ in 1:k
        f()
    end
    w = (time_ns() - t0)/1e9
    c = (ccall(:clock, Culong, ()) - c0)/1e6     # CLOCKS_PER_SEC = 10^6
    c/w
end

#  Benchmarks
const cases = (("ulmBLAS.mul!, strided",          :ulm, true),
               ("LinearAlgebra.mul!, strided",    :jl,  true),
               ("LinearAlgebra.mul!, contiguous", :jl,  false),
               ("ulmBLAS.mul!, contiguous",       :ulm, false))
backend(kind, strided) = kind === :ulm ? "ulmBLAS (1 thread)" :
                         strided ? "generic Julia (1 thread)" : "BLIS"
results = []
println()
@printf("%-31s %-5s %-24s %9s %7s %9s %8s %6s %6s\n", "case", "n", "backend",
        "requested", "ms", "GFLOPS", "CPU/wall", "bytes", "allocs")
for t in 1:4
    setthreads(t)
    @assert getthreads() == t
    for n in (1000, 1500)
        Random.seed!(n)
        P  = rand(2n, n)
        A  = view(P, 1:2:2n, :)
        Ac = Matrix(A)
        B  = rand(n, n)
        C  = zeros(n, n)
        R  = Ac * B                                    # BLIS
        G  = LinearAlgebra.mul!(zeros(n, n), A, B)     # generic Julia
        @assert R ≈ G "n = $n: BLIS and generic Julia differ"
        for (name, kind, str) in cases
            X = str ? A : Ac
            fill!(C, NaN)
            kind === :ulm ? ulmBLAS.mul!(C, X, B) : LinearAlgebra.mul!(C, X, B)
            @assert C ≈ R && C ≈ G "$name, n = $n: wrong result"
            b = kind === :ulm ? @benchmark(ulmBLAS.mul!($C, $X, $B)) :
                                @benchmark(LinearAlgebra.mul!($C, $X, $B))
            tmin = minimum(b).time
            f = kind === :ulm ? () -> ulmBLAS.mul!(C, X, B) :
                                () -> LinearAlgebra.mul!(C, X, B)
            par = cpu_per_wall(f, tmin)
            req = kind === :jl && !str ? string(t) : "-"
            @printf("%-31s %-5d %-24s %9s %7.1f %9.1f %8.2f %6d %6d\n", name, n,
                    backend(kind, str), req, tmin/1e6, 2.0*n^3/tmin, par,
                    b.memory, b.allocs)
            push!(results, (name, n, backend(kind, str), t, tmin,
                            median(b).time, b.memory, b.allocs, length(b.times), par))
        end
    end
end

file = joinpath(@__DIR__, get(ENV, "RESULTS", "gemm_blis") * ".tsv")
open(file, "w") do io
    println(io, "case\tn\tbackend\tBLAS_threads\tmin_ns\tmedian_ns\tbytes\tallocs\tsamples\tcpu_per_wall")
    for r in results
        println(io, join(r, '\t'))
    end
end
