# UlmBLAS.jl: details

## libblastrampoline

- Since Julia 1.7, `LinearAlgebra` calls BLAS and LAPACK through
  [libblastrampoline](https://github.com/JuliaLinearAlgebra/libblastrampoline)
  (LBT), which forwards every routine to a library loaded into it.  By
  default this is OpenBLAS.
- `using UlmBLAS` loads the library of ulmBLAS for this CPU with `dlopen`
  (for `UlmBLAS.mul!`) and does not change LBT.
- `UlmBLAS.forward()` calls
  `BLAS.lbt_forward(UlmBLAS.library_path(); clear=false,
  suffix_hint="64_")`: the routines the library of ulmBLAS provides
  replace those of OpenBLAS, everything else stays with the library that
  provided it before.  `BLAS.get_config()` lists both libraries.  Calling
  it again changes nothing.
- `UlmBLAS.library_path()` is the library in use.

### Routines that ulmBLAS does not provide

LBT keeps one table per interface (ILP64 with suffix `64_`, used by Julia,
and LP64).  `lbt_forward(path; clear=false)` replaces only the entries
for which the new library has a symbol, in the table of its interface; all
other entries keep their previous library (`lbt_forward` in
`src/libblastrampoline.c` of LBT 5.15).  This is how AppleAccelerate.jl
loads Accelerate on top of OpenBLAS.  With `clear=true` LBT would first
reset every entry to a function that only prints "Error: no BLAS/LAPACK
library loaded" and returns without a result, so UlmBLAS has no `clear`
option.

With Julia 1.13 and OpenBLAS (`LinearAlgebra.BLAS.get_config()`):

- ulmBLAS: 149 routines, all of the reference BLAS (LAPACK 3.12.1)
  except `?gemmtr`, in all four precisions.
- OpenBLAS (unchanged): LAPACK; of the routines `LinearAlgebra.BLAS`
  calls, `?gemmt`, `?axpby`, `crot`, `zrot`, `csymv`, `zsymv`, `csyr`,
  `zsyr`, and the CBLAS dot products `cblas_?dot`, `cblas_?dot?_sub` that
  `BLAS.dot`, `dotc` and `dotu` use (ulmBLAS has no CBLAS symbols, so LBT
  leaves them as they are); the BLAS extensions `?gemmtr`, `?omatcopy`,
  `?imatcopy`, `?geadd`, `i?amin`, `?sum`, ... .
- The LP64 table (32-bit integers, e.g. OpenBLAS32.jl) is not changed:
  the library of ulmBLAS has only the ILP64 interface.  With OpenBLAS32
  loaded, LP64 calls keep going to OpenBLAS32.

The tests check this for every symbol of LBT in a fresh process
(`test/subprocess/routing.jl`) and run test files of `LinearAlgebra`
with a counter for calls of routines without implementation
(`test/subprocess/linalg.jl`, as in the tests of MKL.jl and
AppleAccelerate.jl).

### Errors

- An invalid argument to a routine of ulmBLAS (e.g. `lda < m`) is
  reported by `xerbla` on `stderr`, and the routine returns, as in
  OpenBLAS.
- No library (no build for this platform, or `ULMBLAS_LIBRARY` naming a
  file that does not exist): `using UlmBLAS` warns; `UlmBLAS.mul!` and
  `UlmBLAS.forward()` raise an error; the BLAS of Julia is unchanged.

## Threads

ulmBLAS computes each call in the calling thread (it starts no threads of
its own; `BLAS.set_num_threads` does not apply).  For calls of
`UlmBLAS.mul!` and, after `UlmBLAS.forward()`, of BLAS routines from
several Julia threads at the same time:

- Calls with independent inputs and outputs: safe.
- Calls sharing inputs that no call writes (e.g. the same `A` in several
  products): safe.  ulmBLAS never writes its inputs.
- Calls whose output overlaps an input or output of another concurrent
  call: not supported (as in any BLAS); the result is unspecified.

Why (source references: ulmBLAS d738372b):

- The packing buffers of the level 2 and 3 routines come from a cache per
  OS thread (`thread_local` in `MemoryPool::threadCache`,
  `ulmblas/auxiliary/memorypool.tcc:77`).  Every routine takes its blocks
  at the beginning of a call and returns them at the end, on the same
  thread (a Julia task does not change its thread during a `ccall`), with
  no return in between.  A cached block is used by at most one call at a
  time; the cache of a thread is only accessed by that thread and frees
  its blocks when the thread ends.  The `static MemoryPool<T>` objects in
  the routines have no data members.
- Everything else is local to a call: tiles of `C` on the stack, the
  packed operands passed to the micro kernels.  The kernels are selected
  when ulmBLAS is compiled (no detection at run time); the constant
  tables of the vendored AVX-512 kernels are only read.  `xerbla` only
  writes to `stderr`.
- Julia side: the library path and the addresses of
  `ulmblas_?gemm_strided` are set once in `__init__` (while Julia loads
  the package) and only read afterwards; `UlmBLAS.mul!` keeps its
  operands alive during the call (`GC.@preserve`).
- `UlmBLAS.forward()`: libblastrampoline does not synchronize
  `lbt_forward` (its list of loaded libraries, `record_library_load` in
  `config.c` of LBT 5.15, is written without a lock), so UlmBLAS
  serializes its own `forward()` calls with a lock.  A BLAS call in
  another thread while `forward()` runs reads the forwarding table of LBT
  while it is written: formally a data race in LBT; each entry is one
  aligned pointer, so on x86-64 and AArch64 the call goes either to
  OpenBLAS or to ulmBLAS, both correct.  Call `forward()` before
  starting concurrent BLAS work.
- Routines that stay with OpenBLAS rely on OpenBLAS being safe for
  concurrent calls (as Julia does).
- The library is never unloaded (neither Julia nor LBT call `dlclose`),
  which the per-thread caches assume.

`test/subprocess/threads.jl` runs such calls concurrently (first calls
included) and checks the results.

## The library

`make lbt` in ulmBLAS builds `libulmblas_lbt.so` (macOS: `.dylib`):

- the Fortran 77 interface of ulmBLAS with 64-bit integers (ILP64) and
  the symbol suffix `_64_` (`dgemm_64_`), as OpenBLAS's ILP64 build that
  Julia uses;
- no Fortran compiler needed: the Fortran functions (`ddot`, ...) are C
  code;
- on Linux the C++ runtime is linked statically (no dependency on the
  `libstdc++` of the compiler);
- the GEMM with arbitrary strides `ulmblas_{s,d,c,z}gemm_strided`
  (`ulmblas_strided.h` of ulmBLAS).  LBT does not know these names;
  `UlmBLAS.mul!` calls them with `ccall`.  With a library built before the
  strided interface, `UlmBLAS.mul!` raises an error.

## `UlmBLAS.mul!`

- For each operand the pointer of element (1,1), the row stride and the
  column stride are passed.  `transpose` swaps the strides, `adjoint`
  swaps them and sets the conjugation flag (complex types), also nested
  once (`adjoint(transpose(X))` is `conj(X)`).
- As in BLAS, `C` is not read for `β = 0`, and `A`, `B` are not read for
  `α = 0`.
- `UlmBLAS.mul_aliased!` checks `Base.mightalias(C, A)` and
  `Base.mightalias(C, B)`.  On a possible overlap it computes `α A B` into
  a new matrix and then `C = AB + β C`; otherwise it calls `UlmBLAS.mul!`.
  It is also available as `UlmBLAS.mul_iansmba!`.

## Kernel selection

`UlmBLAS.variant()` decides from the CPU:

- AArch64: `sme` on macOS if `sysctl hw.optional.arm.FEAT_SME2` is 1,
  otherwise `neon`;
- x86-64: `zen4` (AMD) or `skylakex` (others) with AVX-512 F and DQ,
  `haswell` with AVX2 and FMA, otherwise `x86_64`.  AMD is recognized by
  the model name of the CPU.

`build/build.sh` builds each variant with `make lbt ARCHFLAGS=...`:

| Variant | `ARCHFLAGS` |
|---|---|
| `neon` | `-DULMBLAS_NO_SME` |
| `sme` | `-mcpu=apple-m4` |
| `x86_64` | `-msse3` |
| `haswell` | `-mavx -mavx2 -mfma`, `ULMBLAS_{D,S,C,Z}GEMM_UKERNEL_openblas_haswell_{4x8,8x4,8x2,4x2}` |
| `skylakex` | `-mavx -mavx2 -mfma -mavx512f -mavx512dq`, D `openblas_skylakex_16x12`, S `openblas_skylakex_16x4`, C `openblas_skylakex_8x2`, Z `blis_zen4_12x4` |
| `zen4` | `-mavx -mavx2 -mfma -mavx512f -mavx512dq`, D `blis_zen4_32x6`, S `openblas_skylakex_16x4`, C `openblas_skylakex_8x2`, Z `blis_zen4_12x4` |

(each kernel as `-DULMBLAS_?GEMM_UKERNEL_<name>`).  On macOS
`MACOSX_DEPLOYMENT_TARGET` defaults to 12.0.

`ULMBLAS_LIBRARY` overrides the choice; `UlmBLAS.variant()` still reports
the variant of the CPU.

## Artifacts

- `build/build.sh SRC OUT` writes, per platform,
  `OUT/lib/VARIANT/libulmblas_lbt.*`,
  `OUT/share/licenses/ulmBLAS/LICENSE` and `OUT/share/ulmBLAS/build.txt`
  (commit, compiler, flags).  Variants: `neon`, `sme` on macOS arm64;
  `x86_64`, `haswell`, `skylakex`, `zen4` on Linux x86-64.
- `julia build/artifacts.jl VERSION DIR=TRIPLET ...` packs each directory
  as `dist/UlmBLAS.vVERSION.TRIPLET.tar.gz` and binds it as artifact
  `ulmblas` in `Artifacts.toml`, with
  `https://github.com/michael-lehn/UlmBLAS.jl/releases/download/vVERSION/`
  as download location.
- When loaded, the package takes `lib/VARIANT/libulmblas_lbt.*` from the
  artifact of its platform.

## Licenses

ulmBLAS and this package are BSD 3-Clause.  Some micro kernels come from
BLIS, AOCL-BLAS and OpenBLAS and keep their own copyright and license
terms.  The license file of ulmBLAS, `share/licenses/ulmBLAS/LICENSE` in
every artifact, reproduces them.
