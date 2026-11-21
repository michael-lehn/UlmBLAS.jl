# UlmBLAS.jl

Use the BLAS of [ulmBLAS](https://github.com/michael-lehn/ulmBLAS) in
Julia.  The package does two things:

- `UlmBLAS.mul!` computes `C = α A B + β C` for matrices with arbitrary
  strides, e.g. every second row of a matrix, without copies.
- `UlmBLAS.forward()` makes ulmBLAS the BLAS of Julia: `A * B`,
  `BLAS.gemm!`, triangular solves, `lu`, `cholesky` etc. then call
  ulmBLAS.  The rest of LAPACK and the few BLAS routines ulmBLAS does not
  have stay with OpenBLAS.

`using UlmBLAS` alone does not change the BLAS of Julia.

```julia
using LinearAlgebra, UlmBLAS

P = rand(1000, 300); A = view(P, 1:2:1000, :);   # every second row
B = rand(300, 200); C = zeros(500, 200);
UlmBLAS.mul!(C, A, B);              # C = A B, GEMM of ulmBLAS, no copies
BLAS.get_config()                   # still OpenBLAS only

UlmBLAS.forward();                  # now ulmBLAS for BLAS, OpenBLAS for the rest
C = rand(500, 300) * B;             # dgemm of ulmBLAS
BLAS.get_config()                   # OpenBLAS and ulmBLAS
```

## Contents

- [Benchmarks: `cholesky`](#benchmarks-cholesky)
- [Installation](#installation)
- [ulmBLAS as the BLAS of Julia](#ulmblas-as-the-blas-of-julia)
- [TODO](#todo)
- [GEMM with arbitrary strides: `UlmBLAS.mul!`](#gemm-with-arbitrary-strides-ulmblasmul)
  - [8-bit integers, Float16 and BFloat16](#8-bit-integers-float16-and-bfloat16)
- [First Benchmarks (old)](#first-benchmarks)
- [Further Ideas](#further-ideas)
- [CPU variants](#cpu-variants)
- [Tests](#tests)
- [License](#license)

## Benchmarks: `cholesky`

`cholesky!` (LAPACK `dpotrf`) of a random symmetric positive definite
matrix in double precision, with Julia's default (OpenBLAS) and with
ulmBLAS after `UlmBLAS.forward()`, on 1 thread and on all cores.  GFLOPS
= n³/3 per second, median of
[BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) samples;
"×" is the time of OpenBLAS divided by the time of ulmBLAS (above 1:
ulmBLAS is faster).  UlmBLAS.jl 0.2.0 (ulmBLAS 0.4.1), Julia 1.13 from
[juliaup](https://github.com/JuliaLang/juliaup) with its OpenBLAS
(0.3.30), November 2026.  Script: [bench/potrf.jl](bench/potrf.jl), all
results: [bench/results](bench/results) (`2026-11-21-*-potrf.*`).

On all cores ulmBLAS is faster from n = 500 on, on all four machines
(from n = 200 on on the x86-64 machines).  On 1 thread it is faster on
the M5 and on Ice Lake for the larger matrices, on par on Zen 4, and
slower on the i5-12500 (see below).

**Apple M5** (4 performance and 6 efficiency cores), variant `sme`:

|    n | OpenBLAS, 1 thread | ulmBLAS, 1 thread |    × | OpenBLAS, 10 threads | ulmBLAS, 10 threads |    × |
|-----:|-----:|------:|-----:|------:|------:|-----:|
|  100 | 20.9 |  18.0 | 0.86 |  18.1 |  19.4 | 1.07 |
|  200 | 33.6 |  31.4 | 0.94 |  43.4 |  36.0 | 0.83 |
|  500 | 45.9 |  82.0 | 1.79 |  93.1 | 109.1 | 1.17 |
| 1000 | 51.8 | 134.2 | 2.59 | 161.3 | 219.1 | 1.36 |
| 2000 | 52.9 | 171.8 | 3.25 | 200.2 | 336.0 | 1.68 |
| 4000 | 52.8 | 203.5 | 3.85 | 223.4 | 376.3 | 1.68 |

**Intel Xeon Platinum 8358** (Ice Lake, 2 × 32 cores), variant
`skylakex`:

|    n | OpenBLAS, 1 thread | ulmBLAS, 1 thread |    × | OpenBLAS, 64 threads | ulmBLAS, 64 threads |    × |
|-----:|-----:|-----:|-----:|------:|------:|-----:|
|  100 | 13.7 | 12.2 | 0.89 |  13.7 |  12.1 | 0.89 |
|  200 | 22.1 | 19.4 | 0.88 |  15.6 |  19.4 | 1.24 |
|  500 | 30.4 | 29.9 | 0.98 |  38.6 |  86.3 | 2.24 |
| 1000 | 34.6 | 48.0 | 1.39 |  85.0 | 231.6 | 2.72 |
| 2000 | 42.6 | 62.5 | 1.47 | 249.3 | 549.3 | 2.20 |
| 4000 | 58.0 | 72.6 | 1.25 | 458.8 | 812.5 | 1.77 |

**AMD EPYC 9454** (Zen 4, 2 × 48 cores), variant `zen4`:

|    n | OpenBLAS, 1 thread | ulmBLAS, 1 thread |    × | OpenBLAS, 96 threads | ulmBLAS, 96 threads |    × |
|-----:|-----:|-----:|-----:|------:|------:|-----:|
|  100 | 15.6 | 15.4 | 0.99 |  15.6 |  15.4 | 0.99 |
|  200 | 24.1 | 23.7 | 0.98 |  12.4 |  23.9 | 1.92 |
|  500 | 35.1 | 36.7 | 1.04 |  37.3 |  73.5 | 1.97 |
| 1000 | 41.4 | 44.5 | 1.07 |  74.6 | 204.0 | 2.74 |
| 2000 | 46.3 | 47.8 | 1.03 | 198.2 | 502.5 | 2.54 |
| 4000 | 51.8 | 50.7 | 0.98 | 388.4 | 697.2 | 1.80 |

**Intel Core i5-12500** (6 cores), variant `haswell`:

|    n | OpenBLAS, 1 thread | ulmBLAS, 1 thread |    × | OpenBLAS, 6 threads | ulmBLAS, 6 threads |    × |
|-----:|-----:|-----:|-----:|------:|------:|-----:|
|  100 | 22.1 | 17.5 | 0.79 |  22.0 |  17.1 | 0.78 |
|  200 | 31.8 | 25.4 | 0.80 |  26.3 |  37.5 | 1.43 |
|  500 | 41.2 | 38.1 | 0.93 |  87.9 | 123.8 | 1.41 |
| 1000 | 46.8 | 40.2 | 0.86 | 146.0 | 203.8 | 1.40 |
| 2000 | 53.4 | 48.0 | 0.90 | 221.9 | 243.8 | 1.10 |
| 4000 | 57.1 | 47.9 | 0.84 | 239.0 | 265.5 | 1.11 |

- How: on 1 thread ulmBLAS factors recursively (`trsm` and `syrk` on
  halves).  On several threads larger matrices are factored with tasks
  on tiles, as in [PLASMA](https://github.com/icl-utk-edu/plasma), but in
  place: no copy of the whole matrix.  The kernels are those of OpenBLAS
  and BLIS (and own ones for SME), chosen by problem size.
- n ≤ 200: ulmBLAS is mostly behind (see [TODO](#todo), small
  matrices).
- i5-12500 on 1 thread: `dtrsm` of the variant `haswell` (AVX2) is slower
  than that of OpenBLAS; this is open.
- The M5 is a MacBook, measured with other programs open; the Ice Lake
  and Zen 4 nodes (bwUniCluster, whole node) and the i5-12500 were idle.
- Julia from Homebrew uses Homebrew's OpenBLAS (0.3.34, OpenMP) instead;
  on the M5 that one is slower on several threads (e.g. 4.3 GFLOPS for n
  = 200 on 10 threads), so ulmBLAS is further ahead there
  (`2026-11-21-m5-homebrew-julia-potrf.*`).

## Installation

```julia
using Pkg
Pkg.add("UlmBLAS")
```

Pkg installs prebuilt libraries of ulmBLAS together with the package
(ulmBLAS_jll), and the package loads the one for the CPU (see
[CPU variants](#cpu-variants)):

- macOS on Apple silicon (macOS 11 or later): `neon` (all Apple silicon)
  and `sme` (Apple M4 and later);
- Linux on x86-64 (glibc 2.17 or later, as Julia; tested on RHEL 8,
  Ubuntu 20.04 and newer): `zen4`, `skylakex`, `haswell`, `x86_64`;
- Windows on x86-64 (MinGW-w64 build): the same as on Linux.

Julia 1.10 or later.  On other platforms the package prints a warning;
`UlmBLAS.mul!` and `UlmBLAS.forward()` then raise an error, and Julia
keeps OpenBLAS.

For development, a library built with ulmBLAS replaces the prebuilt ones:

```sh
git clone https://github.com/michael-lehn/ulmBLAS
cd ulmBLAS
make lbt                    # libulmblas_lbt.dylib (macOS) or .so (Linux)
export ULMBLAS_LIBRARY=$PWD/libulmblas_lbt.dylib
```

`make lbt ARCHFLAGS=...` selects the micro kernels, e.g.
`ARCHFLAGS=-mcpu=apple-m4` for SME2.  `ULMBLAS_VARIANT=haswell` (for
example) selects another of the prebuilt libraries, if the CPU can run it
(`UlmBLAS.variants()`).

## ulmBLAS as the BLAS of Julia

After `UlmBLAS.forward()` the BLAS calls of `LinearAlgebra` go to ulmBLAS:

```julia
using LinearAlgebra, UlmBLAS

UlmBLAS.library_path()              # the library in use
UlmBLAS.forward();
A = rand(ComplexF32, 100, 100); B = rand(ComplexF32, 100, 100);
C = similar(A);
mul!(C, A, B);                      # cgemm of ulmBLAS
BLAS.gemm!('C', 'N', one(ComplexF32), A, B, zero(ComplexF32), C);
println(C ≈ A' * B)                 # should print true
```

- LU and Cholesky (`lu`, `cholesky`, `inv`, `\`) come from ulmBLAS if the
  library has them (`?getrf`, `?potrf`, solves and inverses).  The rest of
  LAPACK (`qr`, `eigen`, ...) comes from OpenBLAS and uses OpenBLAS's BLAS
  internally.
- Routines ulmBLAS does not have (e.g. `BLAS.gemmt!`, `BLAS.axpby!`) and
  the dot products of `LinearAlgebra.BLAS` keep going to OpenBLAS; see
  [docs/details.md](docs/details.md#routines-that-ulmblas-does-not-provide).
- Threads: level 3 (and GEMV, GER for large matrices) run on threads of
  ulmBLAS.  `potrf` (larger matrices) and, from 9 threads on, `getrf` run
  on OpenMP tasks (as PLASMA); `getrf` on fewer threads uses a lookahead on
  the threads of ulmBLAS.  All take the number of
  `BLAS.set_num_threads` (`UlmBLAS.forward()` registers the thread
  functions of ulmBLAS with libblastrampoline and keeps the number Julia
  had before).
- `UlmBLAS.forward()` can be called again (no effect).  Call it before
  starting concurrent BLAS work; see
  [docs/details.md](docs/details.md#threads) for concurrent calls.

Julia calls BLAS through
[libblastrampoline](https://github.com/JuliaLinearAlgebra/libblastrampoline);
`UlmBLAS.forward()` loads the library of ulmBLAS into it, on top of
OpenBLAS.  To undo it, load OpenBLAS again:
`BLAS.lbt_forward(OpenBLAS_jll.libopenblas_path; clear=true)`.  Details
in [docs/details.md](docs/details.md).

## TODO

- Small matrices.  For n up to about 100 these are still slower
  than OpenBLAS, and `lu` and `cholesky` are often called with small
  matrices.  ulmBLAS now picks kernels and block sizes by problem size
  (selection tables); the fixed costs per call and the selection for
  small problems remain.
- `dtrsm` on AVX2 CPUs without AVX-512 (variant `haswell`): slower than
  OpenBLAS's, so `cholesky` on 1 thread is 0.8–0.93 of OpenBLAS there
  (i5-12500, see [Benchmarks](#benchmarks-cholesky)).
- A complete LAPACK, as OpenBLAS has: Reference LAPACK compiled against
  ulmBLAS, with ulmBLAS's own `?getrf`, `?potrf`, ... instead of the
  reference versions.  Then `UlmBLAS.forward()` can replace OpenBLAS
  completely.  Later, own versions where they gain (e.g. tile QR, where
  PLASMA gains most).
- Other element types: `UlmBLAS.mul!` has integers, `Float16` and
  `BFloat16` now.  Next: `lu` and `cholesky` for them, `Double64`, ...
  (for these Julia has no BLAS, only its generic code), and dedicated
  kernels step by step (e.g. FP16 on Sapphire Rapids, BF16 on Zen 4).

## GEMM with arbitrary strides: `UlmBLAS.mul!`

`LinearAlgebra.mul!` calls BLAS only if every matrix has row stride 1.
For other views, e.g. `view(P, 1:2:2n, :)`, it falls back to a generic
Julia loop (no BLAS call).  `UlmBLAS.mul!` passes the strides of Julia's
arrays directly to the GEMM of ulmBLAS:

```julia
using LinearAlgebra, UlmBLAS

n = 100
P = rand(2n, n);
A = view(P, 1:2:2n, :);             # every second row: row stride 2
B = rand(n, n);
C = zeros(n, n);

UlmBLAS.mul!(C, A, B);              # C = A B
println(C ≈ A * B)                  # should print true

C0 = rand(n, n); C = copy(C0);
UlmBLAS.mul!(C, A', B, 2.0, -1.0);  # C = 2 A' B - C
println(C ≈ 2 * A' * B - C0)        # should print true

Z = rand(ComplexF64, n, n); W = rand(ComplexF64, 2n, n);
X = view(W, 2n:-2:1, :);            # negative row stride
Y = zeros(ComplexF64, n, n);
UlmBLAS.mul!(Y, transpose(Z), X');  # Y = Zᵀ Xᴴ
println(Y ≈ transpose(Z) * X')      # should print true
```

- Types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64` and the
  integers `Int8` ... `Int64`, `UInt8` ... `UInt64`, the same for `A`,
  `B` and `C`; `α` and `β` are converted.  Also `Int8`/`UInt8` with an
  `Int32` result, `Float16` and `BFloat16` (see below).  Other types raise
  a `MethodError` (no fallback).
- Integers wrap around as in Julia, `UlmBLAS.mul!(C, A, B) == A * B`.
  With the same type for `A`, `B` and `C` ulmBLAS uses its generic kernel.
- `A`, `B`: strided matrices (`Matrix`, views with ranges, also with
  negative or zero steps) and their `transpose` or `adjoint`.  `C`: a
  strided matrix or its `transpose`.
- No copies, no allocations.
- `UlmBLAS.mul!` is not exported and not a method of `LinearAlgebra.mul!`,
  which keeps its own behavior.

**Contract (not checked):** `C` must not overlap with `A` or `B`, and the
elements of `C` must be distinct (no zero stride, except for a dimension
of length 1).  `A` and `B` may overlap.  If `C` might share memory with
`A` or `B`, use `UlmBLAS.mul_aliased!`:

```julia
Q = rand(2n, n); Q0 = copy(Q);
UlmBLAS.mul_aliased!(view(Q, 1:n, :), view(Q, n+1:2n, :), view(Q, 1:2:2n, :));
println(Q[1:n, :] ≈ Q0[n+1:2n, :] * Q0[1:2:2n, :])  # should print true
```

It computes `α A B` into a temporary matrix if `Base.mightalias` reports a
possible overlap, and is `UlmBLAS.mul!` otherwise.

### 8-bit integers, Float16 and BFloat16

`UlmBLAS.mul!` also multiplies matrices Julia itself multiplies with
generic loops:

- `A`, `B` with element type `Int8` or `UInt8` (each its own), `C` with
  `Int32`: exact modulo 2^32, like Julia's `Int32` arithmetic.
- `A`, `B` with `Float16`, `C` with `Float32` (products and sums in
  `Float32`); with the package BFloat16s also `BFloat16`.

The layouts are the same as above.

```julia
using UlmBLAS, BFloat16s

A = rand(Int8, 1024, 1024); B = rand(Int8, 1024, 1024);
C = zeros(Int32, 1024, 1024);
UlmBLAS.mul!(C, A, B);              # C = A B in Int32

X = BFloat16.(randn(1024, 1024)); Y = BFloat16.(randn(1024, 1024));
Z = zeros(Float32, 1024, 1024);
UlmBLAS.mul!(Z, X, Y);              # Z = X Y in Float32
```

On one core of an Apple M5, n = 1024, in G op/s:

| | Julia | `UlmBLAS.mul!` |
|---|---|---|
| `Int8` (result `Int32`) | 33 | 3447 |
| `Float16` | 13 | 1645 (result `Float32`) |
| `BFloat16` | 0.6 | 1731 (result `Float32`) |
| `Float32` (OpenBLAS) | 115 | 1794 |

These need a library of ulmBLAS that has the functions
(`ulmblas_i8i32gemm_strided`, `ulmblas_sbgemm_strided`,
`ulmblas_shgemm_strided`); with an older one `UlmBLAS.mul!` raises an
error that says so.

## First Benchmarks

Old: these benchmarks are from UlmBLAS.jl 0.1 (ulmBLAS 0.2.0,
single-threaded, one fixed micro kernel per variant).  They do not show
the current state.

- [First Benchmark on my MacBook :-)](benchmark_result/benchmark_001.md)
- [Second Benchmark on my MacBook :-(](benchmark_result/benchmark_002.md)
- [Third Benchmark on my MacBook: Accelerate](benchmark_result/benchmark_003.md)
- [Fourth Benchmark on my MacBook: BLIS](benchmark_result/benchmark_004.md)

## Further Ideas

I explored related approaches in 2016 in
[C++/uBLAS](https://www.mathematik.uni-ulm.de/~lehn/test_ublas/) and
[Blaze](https://www.mathematik.uni-ulm.de/~lehn/test_blaze/session1/page01.html).
The Blaze example in particular evaluates expressions such as
`trans(A) + A` and `2 * B` while packing, rather than first constructing
full temporary matrices.

The current interface already supports matrices with arbitrary row and column
strides. But why stop at matrices whose elements are stored in memory?

One possible next step is to implement the **GEMM frame algorithm in Julia**,
while continuing to use ulmBLAS's optimized micro-kernels. Packing would be
the natural boundary: Julia could read elements from more general matrix
objects, convert them to a supported computation type, and pack them into
the buffers expected by the existing kernels.

Some possibilities worth exploring:

- **Mixed element types.** Allow `A`, `B`, and `C` to have different
  element types, with the computation type determined by `A` and `B`
  (and the scaling factors), rather than by `C`. A different element
  type for `C` is not fundamentally difficult: the existing buffered
  micro-kernel path can accumulate into a tile of the computation type,
  then copy or convert the result into `C`. I already explored and
  tested this approach in the earlier C++ experiments. Mixed integer
  and floating-point operands may be worth considering with explicit
  promotion rules, but **integer × integer should not silently become
  floating-point GEMM**: conversion and accumulation can change the
  numerical result. Supporting genuine integer GEMM would require
  suitable arithmetic semantics and kernels.
- **Matrices computed on demand.** Accept read-only matrix objects that
  implement indexed element access, even when no dense backing storage
  exists. Elements could be computed on demand during packing,
  without materializing the matrix itself.
- **More general matrix expressions.** Explore products involving
  element-wise sums and other expressions evaluated during packing.
  This would need careful treatment of repeated evaluation, aliasing,
  type promotion, and the cost of computing each element; it would not
  automatically make every lazy expression fast.
- **Performance experiments.** Compare Julia-level packing and frame-loop
  overhead against the current native implementation, especially for small
  matrices and expensive element access. Preserve the existing fast path
  for ordinary strided arrays.
- **Beyond BLAS: The Linear Algebra Mapping Problem (LAMP).** Even
  support for arbitrary strides, optimized kernels, and all BLAS
  operations does not solve LAMP: finding an efficient mapping from a
  mathematical linear algebra expression to a sequence of computational
  kernels. This requires reasoning about the entire expression, including
  algebraic transformations, matrix properties, and kernel selection. A
  promising direction is a domain-specific language (DSL) and compiler, as
  demonstrated by [Linnea](https://arxiv.org/abs/1912.12924) by Henrik
  Barthels, Christos Psarras, and Paolo Bientinesi.

I'm currently preparing for job interviews, which means I welcome just about
any distraction. :D Still, these are ideas for future experiments,
**not implemented features or an immediate development plan** for the next
few weeks.

## CPU variants

ulmBLAS selects its instruction set at compile time.  The package
contains one library per kind of CPU and loads the one for the CPU it runs
on (`UlmBLAS.variant()`; `UlmBLAS.variants()` lists all the CPU can run).
Within a library, ulmBLAS picks micro kernel and block sizes by problem
size from a selection table if there is one for the CPU (Apple M5, Intel
Ice Lake, Intel Alder Lake, AMD Zen 4); otherwise it uses the default
kernels of the variant:

| Variant | CPU | Micro kernels (S, D, C, Z) |
|---|---|---|
| `sme` | Apple M4 and later | ulmBLAS SME2 |
| `neon` | Apple silicon (M1 and later) | ulmBLAS NEON |
| `zen4` | AMD with AVX-512 (Zen 4, Zen 5) | OpenBLAS SKYLAKEX (S, C), AOCL zen4 (D, Z) |
| `skylakex` | Intel with AVX-512 | OpenBLAS SKYLAKEX (S, D, C), AOCL zen4 (Z) |
| `haswell` | AVX2 and FMA | OpenBLAS HASWELL (S, C, Z), ulmBLAS AVX (D) |
| `x86_64` | other x86-64 | ulmBLAS SSE (D), reference kernels |

The libraries come from
[ulmBLAS_jll](https://github.com/JuliaBinaryWrappers/ulmBLAS_jll.jl)
(ulmBLAS v0.4.1, built by
[Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil/tree/master/U/ulmBLAS)).
The choice of kernels follows the
[benchmark campaigns](https://github.com/michael-lehn/ulmBLAS/tree/master/benchmark_results)
of ulmBLAS; the flags of each variant are in
[docs/details.md](docs/details.md#kernel-selection).

## Tests

```julia
using Pkg; Pkg.test("UlmBLAS")
```

The tests also run, in fresh processes, a check of every routine of
libblastrampoline before and after `using UlmBLAS` and `UlmBLAS.forward()`,
concurrent calls from several threads, every build of ulmBLAS the CPU can
run, and the BLAS and matrix
product tests of `LinearAlgebra` (about two minutes).  With
`ULMBLAS_TEST_LINALG=1` the LAPACK, triangular, dense and symmetric tests
of `LinearAlgebra` are run as well (several minutes).

## License

BSD 3-Clause, like ulmBLAS.  The libraries contain micro kernels of BLIS,
AOCL-BLAS and OpenBLAS; their licenses are installed with ulmBLAS_jll
(`share/licenses/ulmBLAS/LICENSE`).
