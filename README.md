# ulmBLAS.jl

Use the BLAS of [ulmBLAS](https://github.com/michael-lehn/ulmBLAS) in
Julia.  The package does two things:

- `using ulmBLAS` makes ulmBLAS the BLAS of Julia: `A * B`, `BLAS.gemm!`,
  triangular solves etc. then call ulmBLAS.  LAPACK stays with OpenBLAS.
- `ulmBLAS.mul!` computes `C = α A B + β C` for matrices with arbitrary
  strides, e.g. every second row of a matrix, without copies.

```julia
using LinearAlgebra, ulmBLAS

A = rand(500, 300); B = rand(300, 200);
C = A * B;                          # dgemm of ulmBLAS
BLAS.get_config()                   # ulmBLAS for BLAS, OpenBLAS for LAPACK
```

## Installation

The package is not registered, and there are no prebuilt libraries yet.
Build the library with ulmBLAS:

```sh
git clone https://github.com/michael-lehn/ulmBLAS
cd ulmBLAS
make lbt                    # libulmblas_lbt.dylib (macOS) or .so (Linux)
```

and set `ULMBLAS_LIBRARY` before `using ulmBLAS`:

```sh
export ULMBLAS_LIBRARY=$HOME/ulmBLAS/libulmblas_lbt.dylib
julia --project=/path/to/ulmBLAS.jl     # Pkg.instantiate() once
```

In another environment add the package with
`Pkg.develop(path="/path/to/ulmBLAS.jl")`.

Without `ULMBLAS_LIBRARY` the package looks for a prebuilt library for
this CPU.  As there is none yet, it prints a warning and Julia keeps
OpenBLAS.  A library can also be loaded later with
`ulmBLAS.forward("/path/to/libulmblas_lbt.dylib")`.

`make lbt ARCHFLAGS=...` selects the micro kernels, e.g.
`ARCHFLAGS=-mcpu=apple-m4` for SME2 (see [CPU variants](#cpu-variants)).

## ulmBLAS as the BLAS of Julia

After `using ulmBLAS` the BLAS calls of `LinearAlgebra` go to ulmBLAS:

```julia
using LinearAlgebra, ulmBLAS

ulmBLAS.library_path()              # the library in use
A = rand(ComplexF32, 100, 100); B = rand(ComplexF32, 100, 100);
C = similar(A);
mul!(C, A, B);                      # cgemm of ulmBLAS
BLAS.gemm!('C', 'N', one(ComplexF32), A, B, zero(ComplexF32), C);
println(C ≈ A' * B)                 # should print true
```

- LAPACK (`lu`, `qr`, `eigen`, ...) comes from OpenBLAS and uses OpenBLAS's
  BLAS internally.
- Single-threaded: `BLAS.set_num_threads` has no effect on ulmBLAS.

Julia calls BLAS through
[libblastrampoline](https://github.com/JuliaLinearAlgebra/libblastrampoline);
`using ulmBLAS` loads the library of ulmBLAS into it.  Details in
[docs/details.md](docs/details.md).

## GEMM with arbitrary strides: `ulmBLAS.mul!`

`LinearAlgebra.mul!` calls BLAS only if every matrix has row stride 1.
For other views, e.g. `view(P, 1:2:2n, :)`, it falls back to a generic
Julia loop (no BLAS call).  `ulmBLAS.mul!` passes the strides of Julia's
arrays directly to the GEMM of ulmBLAS:

```julia
using LinearAlgebra, ulmBLAS

n = 100
P = rand(2n, n);
A = view(P, 1:2:2n, :);             # every second row: row stride 2
B = rand(n, n);
C = zeros(n, n);

ulmBLAS.mul!(C, A, B);              # C = A B
println(C ≈ A * B)                  # should print true

C0 = rand(n, n); C = copy(C0);
ulmBLAS.mul!(C, A', B, 2.0, -1.0);  # C = 2 A' B - C
println(C ≈ 2 * A' * B - C0)        # should print true

Z = rand(ComplexF64, n, n); W = rand(ComplexF64, 2n, n);
X = view(W, 2n:-2:1, :);            # negative row stride
Y = zeros(ComplexF64, n, n);
ulmBLAS.mul!(Y, transpose(Z), X');  # Y = Zᵀ Xᴴ
println(Y ≈ transpose(Z) * X')      # should print true
```

- Types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`, the same for
  `A`, `B` and `C`; `α` and `β` are converted.  Other types raise a
  `MethodError` (no fallback).
- `A`, `B`: strided matrices (`Matrix`, views with ranges, also with
  negative or zero steps) and their `transpose` or `adjoint`.  `C`: a
  strided matrix or its `transpose`.
- No copies, no allocations.
- `ulmBLAS.mul!` is not exported and not a method of `LinearAlgebra.mul!`,
  which keeps its own behavior.

**Contract (not checked):** `C` must not overlap with `A` or `B`, and the
elements of `C` must be distinct (no zero stride, except for a dimension
of length 1).  `A` and `B` may overlap.  If `C` might share memory with
`A` or `B`, use `ulmBLAS.mul_aliased!`:

```julia
Q = rand(2n, n); Q0 = copy(Q);
ulmBLAS.mul_aliased!(view(Q, 1:n, :), view(Q, n+1:2n, :), view(Q, 1:2:2n, :));
println(Q[1:n, :] ≈ Q0[n+1:2n, :] * Q0[1:2:2n, :])  # should print true
```

It computes `α A B` into a temporary matrix if `Base.mightalias` reports a
possible overlap, and is `ulmBLAS.mul!` otherwise.

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

ulmBLAS selects its micro kernels at compile time.  The package is meant
to contain one library per kind of CPU and to load the one for the CPU it
runs on (`ulmBLAS.variant()`):

| Variant | CPU | Micro kernels (S, D, C, Z) |
|---|---|---|
| `sme` | Apple M4 and later | ulmBLAS SME2 |
| `neon` | Apple M1 to M3 | ulmBLAS NEON |
| `zen4` | AMD with AVX-512 (Zen 4, Zen 5) | OpenBLAS SKYLAKEX (S, C), AOCL zen4 (D, Z) |
| `skylakex` | Intel with AVX-512 | OpenBLAS SKYLAKEX (S, D, C), AOCL zen4 (Z) |
| `haswell` | AVX2 and FMA | OpenBLAS HASWELL |
| `x86_64` | other x86-64 | ulmBLAS SSE (D), reference kernels |

Platforms: macOS on Apple silicon, Linux on x86-64 (glibc).  The choice
follows the
[benchmark campaigns](https://github.com/michael-lehn/ulmBLAS/tree/master/benchmark_results)
of ulmBLAS; the flags of each variant are in
[docs/details.md](docs/details.md#kernel-selection).

## Building the libraries (maintainers)

```sh
build/build.sh ~/ulmBLAS out-mac        # all variants of this platform
julia build/artifacts.jl 0.1.0 out-mac=aarch64-apple-darwin \
                               out-linux=x86_64-linux-gnu
```

- `build/build.sh SRC OUT [VARIANT ...]` builds
  `OUT/lib/VARIANT/libulmblas_lbt.*` from the ulmBLAS sources in `SRC`,
  with the license and a record of the build (commit, compiler, flags).
- `build/artifacts.jl` packs these directories into `dist/` and writes
  `Artifacts.toml`; the tarballs belong to the release `vVERSION` of this
  repository.

## Tests

```julia
using Pkg; Pkg.test("ulmBLAS")
```

With `ULMBLAS_TEST_LINALG=1` the BLAS and matrix product tests of
`LinearAlgebra` are run with ulmBLAS as well (several minutes).

## License

BSD 3-Clause, like ulmBLAS.  The libraries contain micro kernels of BLIS,
AOCL-BLAS and OpenBLAS; their licenses are part of every artifact
(`share/licenses/ulmBLAS/LICENSE`).
