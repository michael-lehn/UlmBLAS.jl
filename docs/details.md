# ulmBLAS.jl: details

## libblastrampoline

- Since Julia 1.7, `LinearAlgebra` calls BLAS and LAPACK through
  [libblastrampoline](https://github.com/JuliaLinearAlgebra/libblastrampoline)
  (LBT), which forwards every routine to a library loaded into it.  By
  default this is OpenBLAS.
- `using ulmBLAS` (or `ulmBLAS.forward(path)`) calls
  `BLAS.lbt_forward(path; clear=false, suffix_hint="64_")`: the routines
  the library of ulmBLAS provides replace those of OpenBLAS, everything
  else (LAPACK) stays with OpenBLAS.  `BLAS.get_config()` lists both
  libraries.
- `ulmBLAS.forward(path; clear=true)` drops the previous libraries first.
- `ulmBLAS.library_path()` is the library in use.

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
  `ulmBLAS.mul!` calls them with `ccall`.  With a library built before the
  strided interface, `ulmBLAS.mul!` raises an error.

## `ulmBLAS.mul!`

- For each operand the pointer of element (1,1), the row stride and the
  column stride are passed.  `transpose` swaps the strides, `adjoint`
  swaps them and sets the conjugation flag (complex types), also nested
  once (`adjoint(transpose(X))` is `conj(X)`).
- As in BLAS, `C` is not read for `β = 0`, and `A`, `B` are not read for
  `α = 0`.
- `ulmBLAS.mul_aliased!` checks `Base.mightalias(C, A)` and
  `Base.mightalias(C, B)`.  On a possible overlap it computes `α A B` into
  a new matrix and then `C = AB + β C`; otherwise it calls `ulmBLAS.mul!`.
  It is also available as `ulmBLAS.mul_iansmba!`.

## Kernel selection

`ulmBLAS.variant()` decides from the CPU:

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

`ULMBLAS_LIBRARY` overrides the choice; `ulmBLAS.variant()` still reports
the variant of the CPU.

## Artifacts

- `build/build.sh SRC OUT` writes, per platform,
  `OUT/lib/VARIANT/libulmblas_lbt.*`,
  `OUT/share/licenses/ulmBLAS/LICENSE` and `OUT/share/ulmBLAS/build.txt`
  (commit, compiler, flags).  Variants: `neon`, `sme` on macOS arm64;
  `x86_64`, `haswell`, `skylakex`, `zen4` on Linux x86-64.
- `julia build/artifacts.jl VERSION DIR=TRIPLET ...` packs each directory
  as `dist/ulmBLAS.vVERSION.TRIPLET.tar.gz` and binds it as artifact
  `ulmblas` in `Artifacts.toml`, with
  `https://github.com/michael-lehn/ulmBLAS.jl/releases/download/vVERSION/`
  as download location.
- When loaded, the package takes `lib/VARIANT/libulmblas_lbt.*` from the
  artifact of its platform.

## Licenses

ulmBLAS and this package are BSD 3-Clause.  Some micro kernels come from
BLIS, AOCL-BLAS and OpenBLAS and keep their own copyright and license
terms.  The license file of ulmBLAS, `share/licenses/ulmBLAS/LICENSE` in
every artifact, reproduces them.
