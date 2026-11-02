# Netlib LAPACK on top of ulmBLAS

An experiment, not part of the package: Netlib LAPACK 3.12.1, with every
BLAS call executed by ulmBLAS, in Julia, and its LU factorization
(`DGETRF`) compared with Julia's default LAPACK (OpenBLAS).  The package
and its public API are unchanged; everything is in this directory.

## Approach

`UlmBLAS.forward()` replaces only the BLAS routines in libblastrampoline
(LBT).  `LinearAlgebra.LAPACK.getrf!` still runs OpenBLAS's `dgetrf`,
which calls OpenBLAS's kernels directly, not through LBT: ulmBLAS is
never called (shown by `verify.jl`, below).  A LAPACK whose BLAS calls go
through LBT is needed:

```
Julia ──dgetrf_64_──▶ LBT ──▶ Netlib liblapack ──dgemm_64_, dtrsm_64_, ...──▶ LBT ──▶ ulmBLAS
                                  (dgetrf2, dlaswp, ilaenv: inside liblapack)
```

- `build.sh OUT [JULIA]` builds Netlib LAPACK 3.12.1 (release tarball,
  sha256 checked) with CMake as a shared library **linked against the
  libblastrampoline of the Julia installation**.  The only change of the
  sources: `STOP` removed from `xerbla.f` (as in Yggdrasil's LAPACK_jll).
- Configuration in LBT (`common.jl`), on top of OpenBLAS (`clear=false`):
  `UlmBLAS.forward()` for the BLAS routines, then
  `BLAS.lbt_forward(liblapack; clear=false, suffix_hint="64_")` for the
  LAPACK routines.  Everything neither library provides stays with OpenBLAS.
  LBT does provide the separation of BLAS and LAPACK: it forwards each
  symbol separately, and the library loaded last wins for the symbols it
  exports.  Netlib LAPACK exports no BLAS routines (only `lsame` and
  `xerbla`), so the BLAS forwards of ulmBLAS stay in place.
- Three configurations: `openblas` (Julia's default), `netlib_ulmblas`, and
  `netlib_openblas` (the same Netlib LAPACK with OpenBLAS's BLAS, to separate
  the effect of the LAPACK algorithm from that of the BLAS).

**Interface.** Julia uses ILP64 with the suffix `64_`.  The library is
built with LAPACK's extended `_64` API (`BUILD_INDEX64_EXT_API`): every
routine is compiled with `-fdefault-integer-8` and renamed by
`SRC/lapack_64.h`, so gfortran emits `dgetrf_64_` and calls `dgemm_64_`,
`dtrsm_64_`, ... (132 BLAS routines, all imported from LBT, all provided by
ulmBLAS).  The library also contains the LP64 routines without suffix
(part of that build); LBT is told `suffix_hint="64_"` and they are not
used.  The hidden lengths of character arguments (gfortran: `size_t`
after the last argument) pass through LBT unchanged.

**Linkage.** `liblapack` needs `libblastrampoline.so.5`
(`@rpath/libblastrampoline.5.dylib` on macOS, with an rpath to the Julia
used for the build).  At run time this must be the instance Julia has
loaded, otherwise its forwards would be empty; the counters of
`verify.jl` show that the calls go through Julia's LBT.  LAPACK's calls of
its own routines (`dgetrf2_64_`, `dlaswp_64_`, `ilaenv_64_`) are bound
inside `liblapack`, not through LBT (zero calls through LBT; on Linux also
the bindings of the dynamic linker, `bindings.jl` with `LD_DEBUG`).

## Verification of the symbol resolution

`verify.jl` reports, for each configuration, where LBT forwards each BLAS
routine liblapack imports (`nm -u`, `lbt_get_forward`, `dladdr`), and then
replaces the LBT entries of `dgemm`, `dtrsm`, `idamax`, `dscal`, `dswap`,
`dgetrf`, `dgetrf2`, `dlaswp` and `ilaenv` by counters that call the
previous target, for one `DGETRF` with n = 1000.  It fails unless DGEMM
and DTRSM reach ulmBLAS.

Result (Ice Lake and Zen 4, Julia 1.13.3; the same on an Apple M5,
macOS): one `DGETRF` with n = 1000

| through LBT | `openblas` | `netlib_ulmblas` | `netlib_openblas` |
|---|---|---|---|
| `dgetrf` | 1 → OpenBLAS | 1 → liblapack | 1 → liblapack |
| `dgemm`, `dtrsm`, `idamax`, `dscal` | 0 each | 999 each → ulmBLAS | 999 each → OpenBLAS |
| `dgetrf2`, `dlaswp`, `ilaenv`, `dswap` | 0 | 0 | 0 |
| 132 BLAS imports of liblapack | → OpenBLAS | all → ulmBLAS | → OpenBLAS |

DGEMM does 95.4% and DTRSM 4.6% of the flops.  With Julia's default not a
single BLAS call goes through LBT: OpenBLAS's LU uses its own kernels, so
`UlmBLAS.forward()` alone does not change `lu`.  On Linux the dynamic
linker binds `dgemm_64_`, `dtrsm_64_`, `idamax_64_`, `dscal_64_` of
liblapack to Julia's `libblastrampoline.so.5`, and `dgetrf2_64_`,
`dlaswp_64_`, `ilaenv_64_` to liblapack itself
(`results/*.bindings.txt`); `libgfortran.so.5` is Julia's.

## Correctness

`test.jl`: `DGETRF` of 44 matrices: square (n = 1 to 1000, around the
block size 64), rectangular (1×50 to 1000×257), leading dimension larger
than m, matrices with known pivots (A = P·L·U with |L| ≤ 1/2), a pivot in
the last row at every step, zero and all-ones matrices, a zero column at
k = 3 to 300 (INFO = k).  Checked: INFO and the pivots equal those of
OpenBLAS (and the known ones), `i ≤ ipiv[i] ≤ m` (1-based, row i
interchanged with row ipiv[i]), residual
‖P·A − L·U‖₁ / (‖A‖₁ max(m,n) ε) ≤ 30, rows beyond m unchanged, and
`lu(A)` and `A \ b` of LinearAlgebra.

Result: 632 of 632 checks pass for `netlib_ulmblas` and `netlib_openblas`
on Ice Lake, Zen 4 and an Apple M5; the pivots equal those of OpenBLAS in
every case, residuals ≤ 0.05 (`results/*.test.txt`).

## Benchmarks

`bench.jl`: median time of a complete `DGETRF` call (n×n, through LBT as
LinearAlgebra calls it, pivoting and all BLAS calls included), one
thread (`BLAS.set_num_threads(1)`, `OPENBLAS_NUM_THREADS=1`, Julia with one
thread, pinned to one core), the same matrix (randn, seed n) and the same
Julia for all configurations, which take turns for each n.

bwUniCluster 3.0, one exclusive node each (dev partitions, 2026-11-02),
Julia 1.13.3 (OpenBLAS 0.3.30 of Julia), Netlib LAPACK built with
gfortran 14.2 (`-O3`), ulmBLAS of the package's artifact (6c554eec):

- Intel Xeon Platinum 8358 (Ice Lake, 2.6 GHz), ulmBLAS variant `skylakex`
  (`results/2026-11-02-icelake-xeon8358.*`)
- AMD EPYC 9454 (Zen 4), variant `zen4` (`results/2026-11-02-zen4-epyc9454.*`)

![DGETRF time](results/random-time.svg)

![DGETRF speedup](results/random-speedup.svg)

Median time and speedup (time of OpenBLAS / time; below 1 = slower than
Julia's default):

| n | Ice Lake OpenBLAS | Netlib + ulmBLAS | Netlib + OpenBLAS | Zen 4 OpenBLAS | Netlib + ulmBLAS | Netlib + OpenBLAS |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.44 µs | 8.7 µs (0.05×) | 1.2 µs (0.36×) | 0.33 µs | 7.8 µs (0.04×) | 0.81 µs (0.41×) |
| 32 | 5.4 µs | 43 µs (0.12×) | 8.8 µs (0.61×) | 3.9 µs | 41 µs (0.10×) | 5.9 µs (0.67×) |
| 128 | 88 µs | 286 µs (0.31×) | 103 µs (0.85×) | 74 µs | 246 µs (0.30×) | 82 µs (0.89×) |
| 512 | 3.0 ms | 4.7 ms (0.65×) | 3.7 ms (0.83×) | 2.6 ms | 5.4 ms (0.47×) | 4.6 ms (0.56×) |
| 1024 | 18.8 ms | 22.1 ms (0.85×) | 19.4 ms (0.97×) | 17.1 ms | 28.5 ms (0.60×) | 26.3 ms (0.65×) |
| 2048 | 114 ms | 133 ms (0.85×) | 123 ms (0.92×) | 123 ms | 186 ms (0.66×) | 183 ms (0.67×) |
| 4096 | 0.72 s | 0.99 s (0.72×) | 0.94 s (0.76×) | 0.91 s | 1.21 s (0.75×) | 1.24 s (0.74×) |

- Netlib LAPACK on top of ulmBLAS is slower than Julia's default for every
  size: 0.72–0.88× of OpenBLAS for n ≥ 768 on Ice Lake, 0.6–0.8× on Zen 4,
  and much slower for small matrices (0.04–0.3× for n ≤ 128).
- Large n: most of the gap is the LAPACK algorithm, not the BLAS: Netlib
  LAPACK with OpenBLAS's BLAS is nearly as slow.  For n ≥ 1024 ulmBLAS
  makes it 2–8% slower on Zen 4 (2% faster at n = 4096) and 5–14% slower
  on Ice Lake.  Both Netlib configurations drop at n = 256 and 512 (and
  1024 on Zen 4), leading dimensions that are powers of two.
- Small n: the gap is in the BLAS calls.  Netlib's recursive panel
  factorization makes about 4 BLAS calls per column (n = 8: 7 DGEMM,
  7 DTRSM, 8 IDAMAX, 8 DSCAL), and these tiny calls cost about 250 ns
  more each in ulmBLAS than in OpenBLAS (n = 8: 8.7 µs against 1.2 µs with
  the same LAPACK).  The cause within ulmBLAS was not investigated here.

## Reproduce

```sh
sh lapack/build.sh lapack/liblapack julia           # Netlib LAPACK for this Julia
julia --project=lapack -e 'using Pkg; Pkg.instantiate()'
julia --project=lapack lapack/verify.jl
julia --project=lapack lapack/test.jl
julia -t1 --project=lapack lapack/bench.jl results.tsv   # [n ...]
julia lapack/plot.jl OUTDIR "Ice Lake=a.tsv" "Zen 4=b.tsv"
```

On bwUniCluster: `cluster/prepare.sh R JULIA` on a login node (depot and
LAPACK tarball, R/src = these sources), then
`sbatch -p dev_cpu_il cluster/job.sh R JULIA` (build, verify, bindings,
tests, benchmark on one exclusive node).

## Limitations

- An experiment: not wired into the package.  The configuration is global
  to the process (LBT), as for `UlmBLAS.forward()`.
- liblapack must be built against the LBT of the Julia it runs in
  (libblastrampoline 5).  It uses the libgfortran of the compiler it was
  built with (on Linux the soname matches Julia's copy; on macOS
  Homebrew's libgfortran is loaded next to Julia's).
- The algorithms differ: Netlib's `DGETRF` is blocked (NB = 64 from
  `ILAENV`) with the recursive `DGETRF2` for the panels; OpenBLAS has its
  own LU.  `netlib_openblas` separates the two effects.
- Single-threaded only.  The ulmBLAS libraries are those of the package's
  artifact (ulmBLAS 6c554eec), not yet those of ulmBLAS_jll 0.2.
