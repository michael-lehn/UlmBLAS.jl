# First Benchmark on my MacBook :-)

I wanted to see how ulmBLAS performs on my MacBook Air with an
Apple M5, especially when a matrix is not stored contiguously.

Here is what I tested:

- Two square matrix sizes: 1000 × 1000 and 1500 × 1500.
- **Contiguous matrices:** ordinary Julia matrices.
- **Strided matrices:** a view taking every second row of a larger
  matrix. No data is copied to make the view.
- **ulmBLAS:** always one thread.
- **OpenBLAS:** one, two, three, and four threads.
- **Julia's generic multiplication:** what `LinearAlgebra.mul!` uses
  for this particular strided view.

The smaller the time in milliseconds (`ms`), the faster the
multiplication. `GFLOPS` measures billions of floating-point operations
per second, so **higher is better**.

```julia
# Start Julia in the bench environment (BenchmarkTools), with
# ULMBLAS_LIBRARY set:  julia --project=bench
using LinearAlgebra, Libdl, Printf, Random, BenchmarkTools, ulmBLAS
import OpenBLAS_jll

# `using ulmBLAS` routes Julia's BLAS calls to ulmBLAS.  Switch them back to
# OpenBLAS, so that LinearAlgebra.mul! measures OpenBLAS.
oblib  = OpenBLAS_jll.libopenblas_path
ulmlib = ulmBLAS.library_path()
BLAS.lbt_forward(oblib; clear=true, suffix_hint="64_")

# Check: Julia's dgemm is the one of OpenBLAS ...
addr(lib, f) = UInt(dlsym(dlopen(lib), f))
@assert UInt(BLAS.lbt_get_forward("dgemm_", :ilp64)) == addr(oblib, "dgemm_64_")
# ... and ulmBLAS.mul! calls ulmblas_dgemm_strided of ulmBLAS.
@assert UInt(ulmBLAS.gemm_strided_ptr[Float64]) ==
        addr(ulmlib, "ulmblas_dgemm_strided")

# Check: count the dgemm calls of LinearAlgebra.mul! (the strided view
# gets none: Julia uses its generic loop instead of BLAS).
realgemm = BLAS.lbt_get_forward("dgemm_", :ilp64)
ncalls = Ref(0)
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
cf = @cfunction(countgemm, Cvoid,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
                 Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Ptr{Float64},
                 Ptr{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Csize_t,
                 Csize_t))
let P = rand(64, 32), A = view(P, 1:2:64, :), B = rand(32, 32),
    C = zeros(32, 32)
    BLAS.lbt_set_forward("dgemm_", cf, :ilp64)
    ncalls[] = 0; LinearAlgebra.mul!(C, A, B);          s = ncalls[]
    ncalls[] = 0; LinearAlgebra.mul!(C, Matrix(A), B);  c = ncalls[]
    BLAS.lbt_set_forward("dgemm_", realgemm, :ilp64)
    @assert s == 0 && c == 1
end

oblas_threads() = ccall((:openblas_get_num_threads64_, oblib), Cint, ())
gflops(n, t) = 2.0*n^3/t

for threads in 1:4
    BLAS.set_num_threads(threads)            # affects OpenBLAS only
    @assert oblas_threads() == threads
    for n in (1000, 1500)
        Random.seed!(n)
        P  = rand(2n, n)
        A  = view(P, 1:2:2n, :)              # every second row: stride 2
        Ac = Matrix(A)                       # the same matrix, contiguous
        B  = rand(n, n)
        C  = zeros(n, n)
        R  = Ac * B                          # OpenBLAS
        for (name, X, ulm) in (("ulmBLAS.mul!, strided",          A,  true),
                               ("LinearAlgebra.mul!, strided",    A,  false),
                               ("LinearAlgebra.mul!, contiguous", Ac, false),
                               ("ulmBLAS.mul!, contiguous",       Ac, true))
            ulm ? ulmBLAS.mul!(C, X, B) : LinearAlgebra.mul!(C, X, B)
            @assert C ≈ R
            b = ulm ? @benchmark(ulmBLAS.mul!($C, $X, $B)) :
                      @benchmark(LinearAlgebra.mul!($C, $X, $B))
            t = minimum(b).time
            th = !ulm && X === Ac ? threads : 1
            @printf("%-31s %-5d threads %d %8.1f ms %7.1f GFLOPS %3d bytes %d allocs\n",
                    name, n, th, t/1e6, gflops(n, t), b.memory, b.allocs)
        end
    end
end
```

**To reproduce the benchmark**, run the Julia example above (start
Julia with `--project=bench` and set `ULMBLAS_LIBRARY` as described
under [Installation](../README.md#installation)). It is self-contained: it switches
Julia's BLAS calls back to OpenBLAS after `using ulmBLAS`, checks which
implementation is called, verifies the results, and measures execution
time and allocations.

You can also run the [full benchmark script](../bench/gemm_threads.jl),
which records the measurements in a TSV file. The output below is from
the original M5 run; individual timings will vary slightly between runs.

Here is the actual output from my M5. The times are the fastest
measurements reported by BenchmarkTools:

```text
library of ulmBLAS: /Users/lehn/git/ulmBLAS.jl/dev/lib/libulmblas_lbt.dylib
LBT after `using ulmBLAS`: dgemm_ -> ulmBLAS
LBT for this benchmark:   dgemm_ -> OpenBLAS (libopenblas64_.0.3.30.dylib)
ulmBLAS.mul!:             ulmblas_dgemm_strided of libulmblas_lbt.dylib
dgemm_ calls through LBT: LinearAlgebra.mul! strided 0, contiguous 1; ulmBLAS.mul! 0
CPU: Apple M5 (4 performance + 6 efficiency cores), Julia 1.13.3, Julia threads 1, OpenBLAS libopenblas64_.0.3.30.dylib

case                            n     backend                   threads      ms    GFLOPS   bytes allocs
ulmBLAS.mul!, strided           1000  ulmBLAS (1 thread)              1     4.9     408.5       0      0
LinearAlgebra.mul!, strided     1000  generic Julia (1 thread)        1   262.5       7.6      64      1
LinearAlgebra.mul!, contiguous  1000  OpenBLAS                        1    33.8      59.2       0      0
ulmBLAS.mul!, contiguous        1000  ulmBLAS (1 thread)              1     4.6     438.9       0      0
ulmBLAS.mul!, strided           1500  ulmBLAS (1 thread)              1    17.9     378.1       0      0
LinearAlgebra.mul!, strided     1500  generic Julia (1 thread)        1   905.7       7.5      64      1
LinearAlgebra.mul!, contiguous  1500  OpenBLAS                        1   119.6      56.5       0      0
ulmBLAS.mul!, contiguous        1500  ulmBLAS (1 thread)              1    16.3     415.1       0      0
ulmBLAS.mul!, strided           1000  ulmBLAS (1 thread)              1     5.2     386.4       0      0
LinearAlgebra.mul!, strided     1000  generic Julia (1 thread)        1   270.0       7.4      64      1
LinearAlgebra.mul!, contiguous  1000  OpenBLAS                        2    20.1      99.3       0      0
ulmBLAS.mul!, contiguous        1000  ulmBLAS (1 thread)              1     4.7     425.1       0      0
ulmBLAS.mul!, strided           1500  ulmBLAS (1 thread)              1    18.0     374.9       0      0
LinearAlgebra.mul!, strided     1500  generic Julia (1 thread)        1   954.1       7.1      64      1
LinearAlgebra.mul!, contiguous  1500  OpenBLAS                        2    70.4      95.9       0      0
ulmBLAS.mul!, contiguous        1500  ulmBLAS (1 thread)              1    16.0     421.5       0      0
ulmBLAS.mul!, strided           1000  ulmBLAS (1 thread)              1     5.2     384.7       0      0
LinearAlgebra.mul!, strided     1000  generic Julia (1 thread)        1   276.5       7.2      64      1
LinearAlgebra.mul!, contiguous  1000  OpenBLAS                        3    15.2     131.6       0      0
ulmBLAS.mul!, contiguous        1000  ulmBLAS (1 thread)              1     4.8     414.5       0      0
ulmBLAS.mul!, strided           1500  ulmBLAS (1 thread)              1    18.1     373.5       0      0
LinearAlgebra.mul!, strided     1500  generic Julia (1 thread)        1   931.1       7.2      64      1
LinearAlgebra.mul!, contiguous  1500  OpenBLAS                        3    51.5     131.1       0      0
ulmBLAS.mul!, contiguous        1500  ulmBLAS (1 thread)              1    16.1     419.4       0      0
ulmBLAS.mul!, strided           1000  ulmBLAS (1 thread)              1     5.2     383.7       0      0
LinearAlgebra.mul!, strided     1000  generic Julia (1 thread)        1   277.7       7.2      64      1
LinearAlgebra.mul!, contiguous  1000  OpenBLAS                        4    12.0     166.6       0      0
ulmBLAS.mul!, contiguous        1000  ulmBLAS (1 thread)              1     4.8     420.1       0      0
ulmBLAS.mul!, strided           1500  ulmBLAS (1 thread)              1    18.2     371.7       0      0
LinearAlgebra.mul!, strided     1500  generic Julia (1 thread)        1   939.4       7.2      64      1
LinearAlgebra.mul!, contiguous  1500  OpenBLAS                        4    44.9     150.4       0      0
ulmBLAS.mul!, contiguous        1500  ulmBLAS (1 thread)              1    16.4     410.8       0      0
```

This long output says it all:

![GFLOPS of ulmBLAS.mul! and LinearAlgebra.mul! for strided and contiguous matrices, n = 1000 and 1500, 1 to 4 OpenBLAS threads, on an Apple M5](../bench/results/2026-10-31-apple-m5.svg)

## Some notes on that

- **The strided view matters.** Julia's usual `LinearAlgebra.mul!`
  takes about 906 ms for the 1500 × 1500 view. `ulmBLAS.mul!` takes
  about 18 ms for the same operation — roughly **50 times faster**.
- **Strides cost relatively little in ulmBLAS.** At size 1500,
  the strided multiplication takes about 18 ms, compared with about
  16 ms for a contiguous matrix. No contiguous copy of the input
  matrix is needed.
- **More OpenBLAS threads help.** For a contiguous 1500 × 1500
  matrix, OpenBLAS goes from about 120 ms with one thread to
  45 ms with four.
- **ulmBLAS is still faster in this test.** Its single-threaded
  contiguous multiplication takes about 16 ms.
- **No extra allocations in ulmBLAS.** The `bytes` and `allocs`
  columns are both zero for its operations.

The output is arranged in four groups, one for each OpenBLAS thread
setting. Only OpenBLAS uses the extra threads. ulmBLAS and Julia's
generic strided multiplication remain single-threaded.

## One important caveat

This does **not** mean ulmBLAS is always faster than OpenBLAS.
My ulmBLAS build uses the Apple M5's **SME2 matrix instructions**,
while the OpenBLAS 0.3.30 shipped with this Julia installation uses
older **NEON instructions** on this machine. So this is also a
comparison of different CPU instruction sets, not just two libraries.

I'd like to repeat the test with a newer OpenBLAS and with Apple's
Accelerate library.
