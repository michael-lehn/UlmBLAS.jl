"""
    UlmBLAS

Use the BLAS of [ulmBLAS](https://github.com/michael-lehn/ulmBLAS) in Julia.

`using UlmBLAS` loads a shared library of ulmBLAS (64-bit integers,
symbols `dgemm_64_` etc.) into libblastrampoline, the BLAS forwarding layer
of Julia, on top of the libraries already loaded (usually OpenBLAS).  The
BLAS routines of ulmBLAS then replace those of OpenBLAS; LAPACK and every
routine ulmBLAS does not provide (e.g. `?gemmt`, `?axpby`) stay with the
library that provided them before (see `LinearAlgebra.BLAS.get_config()`).

ulmBLAS selects its micro kernels when it is compiled.  The package
therefore contains one library per kind of CPU and picks the one for this
CPU when it is loaded (see [`variant`](@ref)).  The environment variable
`ULMBLAS_LIBRARY` (path of a library built with `make lbt`) overrides the
choice.
"""
module UlmBLAS

using Artifacts, Libdl, LinearAlgebra
using Base.BinaryPlatforms.CPUID

const LIBNAME = Sys.isapple() ? "libulmblas_lbt.dylib" : "libulmblas_lbt.so"

#  path and variant of the library in use (set by __init__ or forward)
const library = Ref{String}("")
const chosen  = Ref{String}("")

#  Apple: SME2 (M4 and later) is reported by sysctl
function has_sme2()
    try
        return strip(read(`sysctl -n hw.optional.arm.FEAT_SME2`, String)) == "1"
    catch
        return false
    end
end

is_amd() = occursin("AMD", Sys.cpu_info()[1].model)
has(feature) = CPUID.test_cpu_feature(feature)

"""
    variant()

The build of ulmBLAS for this CPU:

- `"sme"`: Apple M4 and later (SME2 kernels); `"neon"`: other AArch64;
- `"zen4"`: AMD with AVX-512 (F and DQ: Zen 4, Zen 5); `"skylakex"`:
  Intel with AVX-512 (F and DQ); `"haswell"`: AVX2 and FMA; `"x86_64"`:
  everything else (SSE3).
"""
function variant()
    if Sys.ARCH === :aarch64
        return Sys.isapple() && has_sme2() ? "sme" : "neon"
    elseif Sys.ARCH === :x86_64
        if has(CPUID.JL_X86_avx512f) && has(CPUID.JL_X86_avx512dq)
            return is_amd() ? "zen4" : "skylakex"
        elseif has(CPUID.JL_X86_avx2) && has(CPUID.JL_X86_fma)
            return "haswell"
        else
            return "x86_64"
        end
    end
    error("UlmBLAS: no build for $(Sys.ARCH)")
end

#  the library of variant v in the artifact of this platform (installed by
#  Pkg together with the package; nothing if there is none)
function artifact_library(v)
    toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    isfile(toml) || return nothing
    meta = artifact_meta("ulmblas", toml)
    meta === nothing && return nothing
    path = joinpath(artifact_path(Base.SHA1(meta["git-tree-sha1"])),
                    "lib", v, LIBNAME)
    return isfile(path) ? path : nothing
end

"""
    forward(path)

Load the ulmBLAS library at `path` into libblastrampoline, on top of the
libraries already loaded: only the routines the library provides are
replaced, every other routine keeps its previous implementation
(`BLAS.lbt_forward(path; clear=false)`).  Loading the same library again
changes nothing.

There is deliberately no `clear` option: ulmBLAS has no LAPACK and not all
BLAS extensions, and after `clear=true` libblastrampoline would leave these
routines without an implementation (calls only print an error message and
return without a result).  To return to Julia's default, use
`BLAS.lbt_forward(OpenBLAS_jll.libopenblas_path; clear=true)` (or the
`clear=true` load of another complete library), then `forward` again if
desired.
"""
function forward(path::AbstractString)
    n = BLAS.lbt_forward(path; clear=false, suffix_hint="64_")
    n > 0 || error("UlmBLAS: no BLAS routines forwarded from $path")
    library[] = path
    load_strided(path)
    return n
end

"""
    library_path()

Path of the ulmBLAS library in use.
"""
library_path() = library[]

function __init__()
    path = get(ENV, "ULMBLAS_LIBRARY", "")
    if isempty(path)
        chosen[] = variant()
        lib = artifact_library(chosen[])
        if lib === nothing
            @warn "UlmBLAS: no library for this platform (variant " *
                  "$(chosen[])); set ULMBLAS_LIBRARY to a library built " *
                  "with `make lbt`"
            return
        end
        path = lib
    else
        chosen[] = "ULMBLAS_LIBRARY"
        if !isfile(path)
            @warn "UlmBLAS: ULMBLAS_LIBRARY=$path does not exist; the BLAS " *
                  "of Julia is unchanged"
            return
        end
    end
    forward(path)
end

include("strided.jl")

end # module
