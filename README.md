# UlmBLAS.jl

Use the BLAS of [ulmBLAS](https://github.com/michael-lehn/ulmBLAS) in
Julia.  The package does two things:

- `UlmBLAS.mul!` computes `C = α A B + β C` for matrices with arbitrary
  strides, e.g. every second row of a matrix, without copies.
- `UlmBLAS.forward()` makes ulmBLAS the BLAS of Julia: `A * B`,
  `BLAS.gemm!`, triangular solves etc. then call ulmBLAS.  LAPACK and the
  few BLAS routines ulmBLAS does not have stay with OpenBLAS.

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

Julia 1.9 or later.  On other platforms the package prints a warning;
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
  library has them (`?getrf`, `?potrf`, solves and inverses, with OpenMP
  tasks).  The rest of LAPACK (`qr`, `eigen`, ...) comes from OpenBLAS and
  uses OpenBLAS's BLAS internally.
- Routines ulmBLAS does not have (e.g. `BLAS.gemmt!`, `BLAS.axpby!`) and
  the dot products of `LinearAlgebra.BLAS` keep going to OpenBLAS; see
  [docs/details.md](docs/details.md#routines-that-ulmblas-does-not-provide).
- BLAS is single-threaded.  The LAPACK functions take the number of
  threads of `BLAS.set_num_threads` (`UlmBLAS.forward()` registers them
  with libblastrampoline).
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

- LU and Cholesky (`lu`, `cholesky`) from ulmBLAS (`?getrf`, `?potrf`,
  with OpenMP tasks): comes with the next version of the library.
- Then: small matrices.  For n up to about 100 these are still slower
  than OpenBLAS, and `lu` and `cholesky` are often called with small
  matrices.  For this ulmBLAS has to pick kernels and variants by problem
  size, so the cleanup of its configuration comes first.
- Threads: take the number `BLAS.set_num_threads` gives us.
- More LAPACK, PLASMA style (tasks), C++ templates like the rest:
  - solve: `getrs`, `gesv`, `potrs`, `posv`, `trtrs`, `laswp`, so that
    `A \ b` stays in ulmBLAS
  - inverse: `trtri`, `lauum`, `potri`, `getri` (`inv`)
  - `geqrf` (`qr`).  OpenBLAS has its own versions only of the routines
    above (plus `laed3`); QR comes from Reference LAPACK.  Tile QR is where
    PLASMA gains most.
- Other element types: `Float16`, `BFloat16`, integers (with Julia's
  wraparound), `Double64`, ...  For these Julia has no BLAS, only its
  generic code.
  - First instantiate the ulmBLAS templates with the generic kernels for
    all of them (GEMM, then `lu`, `cholesky`).  Even without dedicated
    kernels the blocking should beat a generic loop.
  - Then dedicated kernels, step by step, with assembly or intrinsics
    (e.g. FP16 on Apple M and Sapphire Rapids, BF16 on Zen 4, VNNI for
    `Int8` on Ice Lake).

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
  `B` and `C`; `α` and `β` are converted.  Other types raise a
  `MethodError` (no fallback).
- Integers wrap around as in Julia, `UlmBLAS.mul!(C, A, B) == A * B`.
  There are no tuned integer kernels yet, ulmBLAS uses its generic one.
  Apple M5, one thread, n = 1000, `A * B` against `UlmBLAS.mul!` (2 means
  twice as fast): `Int64` 3.1, `Int32` 1.6, `Int16` 0.9, `Int8` 0.6.  For
  8 and 16 bit ulmBLAS computes in 32 bit, so Julia is faster there.
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

## First Benchmarks

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

ulmBLAS selects its micro kernels at compile time.  The package contains
one library per kind of CPU and loads the one for the CPU it runs on
(`UlmBLAS.variant()`; `UlmBLAS.variants()` lists all the CPU can run):

| Variant | CPU | Micro kernels (S, D, C, Z) |
|---|---|---|
| `sme` | Apple M4 and later | ulmBLAS SME2 |
| `neon` | Apple silicon (M1 and later) | ulmBLAS NEON |
| `zen4` | AMD with AVX-512 (Zen 4, Zen 5) | OpenBLAS SKYLAKEX (S, C), AOCL zen4 (D, Z) |
| `skylakex` | Intel with AVX-512 | OpenBLAS SKYLAKEX (S, D, C), AOCL zen4 (Z) |
| `haswell` | AVX2 and FMA | OpenBLAS HASWELL |
| `x86_64` | other x86-64 | ulmBLAS SSE (D), reference kernels |

The libraries come from
[ulmBLAS_jll](https://github.com/JuliaBinaryWrappers/ulmBLAS_jll.jl)
(ulmBLAS v0.3.0, built by
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
